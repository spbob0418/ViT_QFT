# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import yaml
from datetime import timedelta
from pathlib import Path
import torch.nn.functional as F
from timm.utils import accuracy, ModelEma
import math


from transformers import ViTForImageClassification
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import torch.nn as nn

import torch.distributed as dist

# from engine_finetune import train_one_step, evaluate
from datasets_original import build_dataset
from losses_original import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator
from token_probe import eval_probe

from checkpointSaver import CheckpointSaver
from rename_key import rename_keys
from z_pertensor_reproduce_fp32 import quant_vision_transformer_pertensor_wg_fp_with_qk_layernorm_CushionCache
import wandb
import socket
import os
import utils

DIST_PORT=7777
WANDB_LOG = os.environ.get("WANDB") 
WANDB_PROJ_NAME = os.environ.get("VERSION")

os.environ['WORLD_SIZE'] = '4'

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--training-steps', default=None, type=int)
    parser.add_argument('--eval-every', default=None, type=int)
    parser.add_argument('--save-cp-every', default=None, type=int)
    parser.add_argument('--accum-steps', default=None, type=int)
    parser.add_argument('--start-step', default=0, type=int)
    parser.add_argument('--register-num', default=0, type=int)
    parser.add_argument('--sched-on-updates', default=False, type=bool)
    parser.add_argument('--transfer-learning', default=False, type=bool)
    parser.add_argument('--sample-data-path', default='', type=str)
    



    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    
    node_rank = getattr(args, "ddp_rank", 0)
    if node_rank==0:
        print(args)
    device_id = getattr(args, "dev_device_id", torch.device("cpu"))
    ode_rank = getattr(args, "ddp_rank", 0)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    # wandb logging

    wandb_project = WANDB_PROJ_NAME
    wandb_run_name = args.model 
    ###scratch training version
    if args.resume: 
        if node_rank == 0:
            wandb_log = True 
            wandb.init(project=wandb_project, name=wandb_run_name, config=args, 
                    resume="must",  # 기존 Run 이어서 진행
                    id="q7jv9kjo" 
            )
        else :
            wandb_log = False
    else : 
        wandb_log = WANDB_LOG 
        if wandb_log :
            wandb.init(project=wandb_project, name=wandb_run_name, config=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

            

    

    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    mixup_active = False
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        register_num = args.register_num,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size, 
        finetune=args.finetune,
    )

    if args.finetune:
        checkpoint = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k").state_dict()

        key_mapping, qkv_weights, qkv_biases = rename_keys(checkpoint, args.transfer_learning)

        new_checkpoint = {}

        for old_key, new_key in key_mapping.items():
            if new_key in model.state_dict():
                new_checkpoint[new_key] = checkpoint[old_key]

        for layer_num in qkv_weights.keys():
            if "query" in qkv_weights[layer_num] and "key" in qkv_weights[layer_num] and "value" in qkv_weights[layer_num]:
                qkv_weight = torch.cat([
                    qkv_weights[layer_num]["query"],
                    qkv_weights[layer_num]["key"],
                    qkv_weights[layer_num]["value"]
                ], dim=0)
                new_checkpoint[f"blocks.modules_list.{layer_num}.attn.qkv.weight"] = qkv_weight

        for layer_num in qkv_biases.keys():
            if "query" in qkv_biases[layer_num] and "key" in qkv_biases[layer_num] and "value" in qkv_biases[layer_num]:
                qkv_bias = torch.cat([
                    qkv_biases[layer_num]["query"],
                    qkv_biases[layer_num]["key"],
                    qkv_biases[layer_num]["value"]
                ], dim=0)
                new_checkpoint[f"blocks.modules_list.{layer_num}.attn.qkv.bias"] = qkv_bias

        pos_embed_checkpoint = new_checkpoint['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]

        old_num_extra_tokens = 1
        new_num_extra_tokens = 1 + args.register_num  

        old_cls_token = pos_embed_checkpoint[:, :old_num_extra_tokens, :]   
        old_patch_tokens = pos_embed_checkpoint[:, old_num_extra_tokens:, :] 
        
        with torch.no_grad():
            extra_tokens_from_model = model.pos_embed[:, old_num_extra_tokens:new_num_extra_tokens, :]

        num_patches = model.patch_embed.num_patches  
        orig_size = int((old_patch_tokens.shape[1]) ** 0.5)  
        new_size = int(num_patches ** 0.5)               

        patch_tokens_2d = old_patch_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        patch_tokens_2d = F.interpolate(patch_tokens_2d, size=(new_size, new_size), mode='bicubic', align_corners=False)
        patch_tokens_interp = patch_tokens_2d.permute(0, 2, 3, 1).flatten(1, 2) 

        new_pos_embed = torch.cat(
            [old_cls_token, extra_tokens_from_model, patch_tokens_interp],
            dim=1
        )

        # 7. 체크포인트를 수정하여 새 pos_embed로 덮어쓰기
        new_checkpoint['pos_embed'] = new_pos_embed
        model.load_state_dict(new_checkpoint, strict=False)

    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if node_rank == 0:
        print('number of params:', n_parameters)
    
    if wandb_log:
        wandb.watch(model)
        
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    
    loss_scaler = NativeScaler()
    
    if args.epochs is None: 
        print("training steps", args.training_steps)
        print("data length", len(data_loader_train))
        args.epochs = int(args.training_steps // len(data_loader_train))
    # args.eval_every = len(data_loader_train)
    args.eval_every = 200
    # args.updates_per_epoch = len(data_loader_train)
    # args.decay_epochs = args.epochs - args.warmup_epochs
    # args.decay_milestones
   
    lr_scheduler, _ = create_scheduler(args, optimizer, updates_per_epoch = len(data_loader_train))

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            model=args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_step = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_step)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    saver = None
    if node_rank==0:
        # print(f"Start training for {args.epochs} epochs")
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=False)

        
    start_time = time.time()
    max_accuracy = 0.0
    training_step = 0
    dist.barrier()
    data_iterator = iter(data_loader_train)
    
    # ########################################################################
    # # 동결(Freeze)할 파라미터 설정
    # for name, param in model.named_parameters():
    #     if name in ["module.cls_token", "module.pos_embed", "module.reg_token", 
    #                 "module.head.weight", "module.head.bias"]:
    #         param.requires_grad = True  # 학습 가능하게 설정
    #     else:
    #         param.requires_grad = False  # Freeze (학습 불가능)

    # ########################################################################
    # save_cp_point = [2500, 1700, 700, 3700]
 
    for training_step in range(args.start_step, args.training_steps):
        model.train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Step: [{training_step}]'

        try:
            samples, targets = next(data_iterator)  # 미리 만든 이터레이터에서 한 배치 가져오기
        except StopIteration:
            # 데이터가 끝나면 다시 반복하도록 이터레이터를 새로 생성    
            data_iterator = iter(data_loader_train)
            samples, targets = next(data_iterator)
    
        # for iteration, (samples, targets) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        
        outputs = model(samples, epoch=None, iteration=training_step, device_id=device_id)
    
        loss = criterion(samples, outputs, targets)
        dist.barrier()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()

        if (training_step + 1) % args.accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step_update(training_step)

        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if model_ema is not None:
            model_ema.update(model)
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        
        if wandb_log:
            wandb.log({
                "training_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "training_step": training_step
            })
        metric_logger.synchronize_between_processes()
        if device_id == 0 and training_step % 100 == 0: 
            print(f"Steps [{training_step}/{args.training_steps}] ", metric_logger)
            

        if training_step % (args.accum_steps*50) == 0 and device_id == 0:
            eval_probe(model, training_step, args)

        if training_step % args.eval_every == 0:
        # if training_step in save_cp_point:
            criterion_val = torch.nn.CrossEntropyLoss()
            metric_logger_val = utils.MetricLogger(delimiter="  ")
            header = 'Test:'
            model.eval()
            with torch.no_grad():
                for images, target in metric_logger_val.log_every(data_loader_val, 100, header):
                    images = images.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    
                    output = model(images)
                    loss = criterion_val(output, target)
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    
                    batch_size = images.shape[0]
                    metric_logger_val.update(loss=loss.item())
                    metric_logger_val.meters['acc1'].update(acc1.item(), n=batch_size)
                    metric_logger_val.meters['acc5'].update(acc5.item(), n=batch_size)
                
                metric_logger_val.synchronize_between_processes()
                if device_id == 0:
                    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                        .format(top1=metric_logger_val.acc1, top5=metric_logger_val.acc5, losses=metric_logger_val.loss))
                
                if wandb_log:
                    wandb.log({"Acc@1": metric_logger_val.acc1.global_avg})
                    wandb.log({"Acc@5": metric_logger_val.acc5.global_avg})
                    wandb.log({"val_loss/step": metric_logger_val.loss.global_avg})

                if device_id == 0:
                    print(f"Accuracy of the network on the {len(dataset_val)} test images: {metric_logger_val.acc1.global_avg:.3f}%")
                
                # if device_id == 0:
                #     if max_accuracy < metric_logger_val.meters['acc1'].global_avg:
                #         max_accuracy = metric_logger_val.meters['acc1'].global_avg
                #         if args.output_dir:
                #             checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                #             for checkpoint_path in checkpoint_paths:
                #                 utils.save_on_master({
                #                     'model': model_without_ddp.state_dict(),
                #                     'optimizer': optimizer.state_dict(),
                #                     'lr_scheduler': lr_scheduler.state_dict(),
                #                     'training_step': 'training_step',
                #                     'scaler': loss_scaler.state_dict(),
                #                     'args': args,
                #                 }, checkpoint_path)

       
        
        if (training_step+0) % args.save_cp_every == 0 and saver is not None and device_id == 0:
        # if training_step in save_cp_point and saver is not None and device_id == 0:
            save_metric = metric_logger_val.meters['acc1'].global_avg
            best_metric, best_epoch = saver.save_checkpoint(epoch=training_step+1, metric=save_metric)
        
        torch.cuda.empty_cache()
        dist.barrier()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
        
    if wandb_log:
        wandb.finish()


def distributed_init(args) -> int:
    ddp_url = getattr(args, "ddp_dist_url", None)
    node_rank = int(os.environ["RANK"])  # RANK 환경 변수 사용
    world_size = int(os.environ["WORLD_SIZE"])  # WORLD_SIZE 환경 변수 사용

    if ddp_url is None:
        ddp_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        setattr(args, "ddp_dist_url", ddp_url)

    if torch.distributed.is_initialized():
        print("DDP is already initialized!")
    else:
        print(f"Distributed init (rank {node_rank}): {ddp_url}")
        dist_backend = getattr(args, "ddp_backend", "nccl")
        dist.init_process_group(
            backend=dist_backend,
            timeout=timedelta(seconds=7200000),
            init_method=ddp_url,
            world_size=world_size,
            rank=node_rank,
        )

        # Perform a dummy all-reduce to initialize NCCL
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    setattr(args, "ddp_rank", node_rank)
    return node_rank


def main_worker(local_rank, args):
    setattr(args, "dev_device_id", local_rank)
    torch.cuda.set_device(local_rank)
    setattr(args, "dev_device", torch.device(f"cuda:{local_rank}"))

    # Initialize distributed training
    distributed_init(args)

    # Call the main function
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DeiT training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # LOCAL_RANK provided by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    main_worker(local_rank, args)


