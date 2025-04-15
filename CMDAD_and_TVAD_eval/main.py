import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from collections import OrderedDict

from model.model_ad import VideoCaptionModel

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from dataloader.MAD_load import get_data
from utils.distributed import is_master, init_distributed_device, broadcast_object
from utils.logger import setup_logging
from utils.params import parse_args
from utils.scheduler import cosine_lr, const_lr, const_lr_cooldown
from utils.eval import evaluate
from utils.file_utils import pt_load, check_exists, start_sync_process, remote_sync


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=217, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)
    args.logs = args.mylogs
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    print(args.rank)
    # print('hello')

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{args.LLM_name}",
            f"dataset_{args.dataset}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        args.generated_ad_path = os.path.join(log_base_path, 'generated_ads')
        # if os.path.exists(args.log_path) and not resume_latest:
        #     print(
        #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
        #     )
        #     return -1

    # # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # # Setup wandb, tensorboard, checkpoint logging
    # args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path, args.generated_ad_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 217)
    
    # creat model here
    model = VideoCaptionModel(args)

    # if is_master(args):
    #     print(model.bos_ad_token_id, model.pad_token_id, model.image_token_id, model.img_start_token_id, model.img_end_token_id, model.video_token_id, model.video_start_token_id, model.video_end_token_id)
    
    if args.WebVid_pretrained == 1:
        webvid_checkpoint = torch.load(args.WebVid_pretrained_checkpoint, map_location='cpu')
        webvid_checkpoint_no_module = OrderedDict()
        for k, v in webvid_checkpoint["state_dict"].items():
            if k[:7] == 'module.':
                name = k[7:]    # remove `module.`
            else:
                name = k
            assert 'gpt.' not in name
            webvid_checkpoint_no_module[name] = v

        model.load_state_dict(webvid_checkpoint_no_module, strict=False)
        if is_master(args):
            print('load WebVid_pretrained:', webvid_checkpoint_no_module.keys())

        if args.if_only_flamingo != 1 and args.if_share_img_video == 0:
            webvid_checkpoint_no_module_perceiver_img = OrderedDict()
            for k, v in webvid_checkpoint_no_module.items():
                if k[:10] == 'perceiver.':
                    name = k[10:]
                    webvid_checkpoint_no_module_perceiver_img[name] = v
            model.perceiver_img.load_state_dict(webvid_checkpoint_no_module_perceiver_img, strict=False)
            if is_master(args):
                print('perceiver img load WebVid_pretrained:', webvid_checkpoint_no_module_perceiver_img.keys())
            
    if args.AD_pretrained == 1:
        assert args.if_lora == 1
        ad_checkpoint = torch.load(args.AD_pretrained_checkpoint, map_location='cpu')
        ad_checkpoint_no_module = OrderedDict()
        for k, v in ad_checkpoint["state_dict"].items():
            if k[:7] == 'module.':
                name = k[7:]    # remove `module.`
            else:
                name = k
            if 'gpt.' in name:
                ad_checkpoint_no_module[name] = v
        
        model.load_state_dict(ad_checkpoint_no_module, strict=False)
        # model.ad_input_token_embedding.weight.requires_grad = False
        if is_master(args):
            print('load AD_pretrained:', ad_checkpoint_no_module.keys())

    if args.if_finutune_GPT == 0:
        for param in model.gpt.named_parameters():
            if 'gated_cross_attn_layers' not in param[0]:
                param[1].requires_grad = False
    else:
        assert args.if_lora == 1
        for param in model.gpt.named_parameters():
            if 'lora_' in param[0] or 'norm' in param[0] or 'gated_cross_attn_layers' in param[0]:
                param[1].requires_grad = True
            else:
                param[1].requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'total params: {total_params}')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {trainable_params}')

    model.to(device)
    random_seed(args.seed, args.rank)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'
        if args.Visual_Loss:
            decoder = torch.nn.ModuleList([model.reg_head])
            perceiver = torch.nn.ModuleList([model.perceiver, model.project])
            decoder_params = list(map(id, decoder.parameters()))
            perceiver_params = list(map(id, perceiver.parameters()))
            optimizer = torch.optim.AdamW(
                [
                    {'params': filter(lambda p: p.requires_grad and id(p) not in decoder_params and id(p) not in perceiver_params, model.parameters())},
                    {'params': perceiver.parameters(), 'lr': args.lr_perceiver},
                    {'params': decoder.parameters(), 'lr': args.lr_decoder},
                ],
                lr=args.lr, weight_decay=0.0)
        else:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.0)
        
        scaler = GradScaler() if args.precision == "amp" else None # None

    # optionally resume from a checkpoint
    start_epoch = 0
    if_load = False
    
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if_load = True
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            logging.info(f"load model: '{sd.keys()}' (epoch {start_epoch})")
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint, strict=False)
            logging.info(f"load model: '{checkpoint.keys()}' (epoch {start_epoch})")
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    data = get_data(args, (None, None), epoch=start_epoch, tokenizer=model)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            if args.Visual_Loss:
                scheduler = cosine_lr(optimizer, [args.lr, args.lr_perceiver, args.lr_decoder], args.warmup, total_steps)
            else:
                scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if is_master(args):
        for i, param in enumerate(model.named_parameters()):
            print(i, param[0], param[1].requires_grad)

        evaluate(model, data, start_epoch, args, writer)
    
if __name__ == "__main__":
    main(sys.argv[1:])
