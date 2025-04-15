import json
import logging
import math
import os
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

from .rouge import Rouge
from .cider import Cider
from .ptbtokenizer import PTBTokenizer

try:
    import wandb
except ImportError:
    wandb = None


from .distributed import is_master
from .precision import get_autocast, get_input_dtype
from .gpt_utils import generate_beam
# from .recall_with_neighbors import recall_within_neighbours

from transformers.generation.configuration_utils import GenerationConfig
GENERATION_CONFIG = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        assert 0
    device = torch.device(args.device)
    model.eval()

    fail_num = 0

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    generate_ads_dict = {}
    gt_ads_dict = {}

    rouge_L = Rouge()
    cider = Cider()
    tokenizer = PTBTokenizer()

    generate_ads_path = os.path.join(args.generated_ad_path, args.eval_data_name + '_ad_output_results.json')
    gt_ads_path = os.path.join(args.generated_ad_path, args.eval_data_name + '_ad_output_gts.json')
    reuslt_output = os.path.join(args.generated_ad_path, args.eval_data_name + '_eval.txt')
            
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        for k, v in data['val'].items():
            dataloader = v.dataloader
            num_samples = 0
            samples_per_val = dataloader.num_samples
            gt_ads = []
            generate_ads = []
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    video = batch['video']
                    if video is None:
                        fail_num += 1
                        continue
                    video = video.to(device=device, dtype=input_dtype, non_blocking=True)
                    prompt = batch['prompt'].to(device=device, non_blocking=True)
                    char_image = batch['char_image']
                    idx = batch['image_idx']
                    video_idx = batch['video_idx']
                    if char_image is not None:
                        char_image = char_image.to(device=device, dtype=input_dtype, non_blocking=True)
                        
                    ad = batch['ad']
                    mask = batch['mask'].to(device=device, non_blocking=True)

                    for gt_ad in ad:
                        gt_ads.append(gt_ad)

                    with autocast():
                        GENERATION_CONFIG.pad_token_id = model.pad_token_id
                        GENERATION_CONFIG.bos_token_id = model.tokenizer.bos_token_id
                        GENERATION_CONFIG.eos_token_id = model.tokenizer.eos_token_id
                        _, inputs_embeds, vis_x = model(video_embeds=video, text_input=prompt, charactor_images=char_image, text_mask=mask, idx=idx, video_idx=video_idx)
                        generate_ids = model.gpt.generate(
                            generation_config=GENERATION_CONFIG,
                            inputs_embeds=inputs_embeds,
                            attention_mask=mask,
                            # top_p=0.9,
                            temperature=1,
                            num_beams=5,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            max_length=10000,
                            return_dict=True,
                            lm_embedding_new=[model.video_input_token_embedding, model.image_input_token_embedding],
                            lm_heads_new=[model.video_output_token_embedding, model.image_output_token_embedding],
                            video_length=model.video_token_length,
                            automodel=model,
                            vis_x=vis_x,
                            )
                        generate_ids = generate_ids.masked_fill(
                            generate_ids >= model.num_embeddings, 0
                        )
                        output = model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        batch_size = video.shape[0]

                    num_samples += batch_size
                    for g_ad in output:
                        generate_ads.append(g_ad)
                        
                    if is_master(args) and (i % 10) == 0:
                        with open(reuslt_output, 'a+') as f:
                            f.write(f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t \n")
                
                generate_ads_dict[k] = generate_ads
                gt_ads_dict[k] = gt_ads

        with open(generate_ads_path, 'w') as f:
            json.dump(generate_ads_dict, f, indent=2)

        with open(gt_ads_path, 'w') as f:
            json.dump(gt_ads_dict, f, indent=2)


        all_gts = []
        all_generated = []
        print('fail:', fail_num)
        
        for k, v in gt_ads_dict.items():
            gts = v
            res = generate_ads_dict[k]
            
            gts = tokenizer.tokenize(gts, args)
            res = tokenizer.tokenize(res, args)

            if gts == [''] or res == ['']:
                continue
            
            all_gts.extend(gts)
            all_generated.extend(res)
            
        rouge_l = rouge_L.compute_score(all_gts, all_generated) * 100
        cider_score = cider.compute_score(all_gts, all_generated) * 100
        metrics.update({"Rouge_L": rouge_l})
        metrics.update({"Cider": cider_score})

    if not metrics:
        return metrics

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(reuslt_output, 'a+') as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics, rouge_l + cider_score
