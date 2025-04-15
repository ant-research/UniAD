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

def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    
    losses_m = {}
    para = args.param
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        videos = batch['video']
        texts = batch['text']
        images = batch['char_image']
        idx = batch['image_idx']
        video_idx = batch['video_idx']
        ad_start_ids = batch['ad_start_ids']
        
        texts_neg = batch['text_neg']
        texts_neg = texts_neg.to(device=device, non_blocking=True)

        if videos is None:
            continue
        videos = videos.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        if images is not None:
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            
        with autocast():
            output, _, _ = model(video_embeds=videos, text_input=texts, charactor_images=images, idx=idx, video_idx=video_idx, ad_start_ids=ad_start_ids)
            
            output_neg, _, _ = model(video_embeds=videos, text_input=texts_neg, charactor_images=images, idx=idx, video_idx=video_idx, ad_start_ids=ad_start_ids)
            language_score_neg = output_neg[0] * -1.
            language_score = output[0]* -1.
            contrastiv_loss = max(language_score_neg - language_score, torch.tensor(0.))

            if args.Visual_Loss:
                losses = {
                    "language_loss" : output[0],
                    "visual_loss" : output[-1],
                }
                
                total_loss = (1 - para) * losses["language_loss"] + para * losses["visual_loss"]
            else:
                losses = {
                    "language_loss" : output[0],
                    "contrastiv_loss" : contrastiv_loss,
                }
                total_loss = losses["language_loss"] + losses["contrastiv_loss"]
            
            losses["loss"] = total_loss
            total_loss = total_loss / args.accum_freq

        backward(total_loss, scaler)

        if ((i + 1) % args.accum_freq == 0) or (i + 1 == len(dataloader)):
            if scaler is not None:
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()
            optimizer.zero_grad()

        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(videos)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            # samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            # samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            if args.Visual_Loss:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    # f"Data (t): {data_time_m.avg:.3f} "
                    # f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"LR_Perceiver: {optimizer.param_groups[1]['lr']:5f} "
                    f"LR_Decoder: {optimizer.param_groups[2]['lr']:5f} "
                    f"Logit Scale: " + loss_log
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    # f"Data (t): {data_time_m.avg:.3f} "
                    # f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: " + loss_log
                )
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            if args.Visual_Loss:
                log_data = {
                    # "data_time": data_time_m.val,
                    # "batch_time": batch_time_m.val,
                    # "samples_per_second": samples_per_second,
                    # "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_perceiver": optimizer.param_groups[1]["lr"],
                    "lr_decoder": optimizer.param_groups[2]["lr"],
                }            
            else:
                log_data = {
                    # "data_time": data_time_m.val,
                    # "batch_time": batch_time_m.val,
                    # "samples_per_second": samples_per_second,
                    # "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": optimizer.param_groups[0]["lr"],
                }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            # batch_time_m.reset()
            # data_time_m.reset()
    # end for

def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        assert 0
    device = torch.device(args.device)
    model.eval()

    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    # metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    generate_ads_dict = {}
    gt_ads_dict = {}

    rouge_L = Rouge()
    cider = Cider()
    tokenizer = PTBTokenizer()

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        for k, v in data['val'].items():
            dataloader = v.dataloader
            num_samples = 0
            samples_per_val = dataloader.num_samples
            gt_ads = []
            generate_ads = []
            # generate_ads_greedy = []
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    video = batch['video']
                    if video is None:
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
                        _, inputs_embeds, vis_x = model(video_embeds=video, text_input=prompt, charactor_images=char_image, text_mask=mask, if_train=False, idx=idx, video_idx=video_idx)
                        generate_ids = model.gpt.generate(
                            generation_config=GENERATION_CONFIG,
                            inputs_embeds=inputs_embeds,
                            attention_mask=mask,
                            # top_p=0.9,
                            temperature=1,
                            num_beams=5,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            max_length=67,
                            return_dict=True,
                            lm_embedding_new=[model.video_input_token_embedding, model.image_input_token_embedding],
                            lm_heads_new=[model.video_output_token_embedding, model.image_output_token_embedding],
                            video_length=model.video_token_length,
                            if_train=False,
                            automodel=model,
                            # vis_x=vis_x,
                            )
                        generate_ids = generate_ids.masked_fill(
                            generate_ids >= model.num_embeddings, 0
                        )
                        output = model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        batch_size = video.shape[0]
                
                    num_samples += batch_size
                    for g_ad in output:
                        generate_ads.append(g_ad)
                        # print(g_ad)
                    if is_master(args) and (i % 10) == 0:
                        logging.info(
                            f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                            )
                
                generate_ads_dict[k] = generate_ads
                gt_ads_dict[k] = gt_ads


            generate_ads_path = os.path.join(args.generated_ad_path, 'ad_output_results_epoch_{}.json'.format(epoch))
            gt_ads_path = os.path.join(args.generated_ad_path, 'ad_output_gts.json')
            with open(generate_ads_path, 'w') as f:
                json.dump(generate_ads_dict, f, indent=2)

            with open(gt_ads_path.format(epoch), 'w') as f:
                json.dump(gt_ads_dict, f, indent=2)


        all_gts = []
        all_generated = []
        for k, v in gt_ads_dict.items():
            gts = v
            res = generate_ads_dict[k]
            gts = tokenizer.tokenize(gts, args)
            res = tokenizer.tokenize(res, args)
            all_gts.extend(gts)
            all_generated.extend(res)

        rouge_l = rouge_L.compute_score(all_gts, all_generated) * 100
        cider_score = cider.compute_score(all_gts, all_generated) * 100
        metrics.update({"Rouge_L": rouge_l})
        metrics.update({"Cider": cider_score})
        logging.info(
            f"Rouge_L: {rouge_l}\t"
            )
        logging.info(
            f"Cider: {cider_score}\t"
            )
        
    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics, rouge_l + cider_score
