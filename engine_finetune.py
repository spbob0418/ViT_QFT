
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import csv

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses_original import DistillationLoss
import utils
import torch.distributed as dist
import wandb
from probe import probe
import re
import numpy as np
import os
from optimizer_probe import compute_adamw_update
import pandas as pd
import torch.nn.functional as F

def train_one_step(wandb_log, device_id, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, training_step: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Step: [{}]'.format(training_step)
    
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    for iteration, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples, samples), dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        
        outputs = model(samples, training_step, iteration, device_id)
  

        dist.barrier()
        
        if not args.cosub:
            loss = criterion(
                    samples, 
                    outputs, 
                    targets
                )
        else:
            outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
            loss = 0.25 * criterion(outputs[0], targets) 
            loss = loss + 0.25 * criterion(outputs[1], targets) 
            loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
            loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 
        
        dist.barrier()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()

        if (iteration + 1) % args.accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if wandb_log: 
            wandb.log({
                "training_loss": loss_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_step": training_step
            })

    metric_logger.synchronize_between_processes()
    if device_id == 0: 
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, wandb_log):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    if wandb_log:
        wandb.log({"Acc@1": metric_logger.acc1.global_avg})
        wandb.log({"Acc@5": metric_logger.acc5.global_avg})
        wandb.log({"val_loss/step": metric_logger.loss.global_avg})
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
