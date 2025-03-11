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


def train_one_epoch(wandb_log, device_id, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    ##로그 frequency
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    for iteration, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        
        #with torch.amp.autocast("cuda"):
        outputs = model(samples, epoch, iteration, device_id)
        dist.barrier()
        if not args.cosub:
            
            # loss = criterion(samples, outputs, targets)
            loss = criterion(
                    samples.to(dtype=torch.float32), 
                    outputs.to(dtype=torch.float32), 
                    targets.to(dtype=torch.float32)
                )
            
        else:
            outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
            loss = 0.25 * criterion(outputs[0], targets) 
            loss = loss + 0.25 * criterion(outputs[1], targets) 
            loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
            loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 
        dist.barrier()
        ######

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


        torch.cuda.empty_cache()


        ########################################

        # if device_id == 0:
        #     for name, param in model.named_parameters():
        #         print(name, id(name))
        # exit()
        # if device_id == 0: 
        #     for name, param in model.named_parameters():
        #         if not param.requires_grad:  # requires_grad가 False인 파라미터만 출력
        #             print(name, id(name))
        # exit()
        
        if (device_id == 0) and (iteration % 10000 == 0):
            os.makedirs("./optimizer_state_report_", exist_ok=True)
            state_dict = optimizer.state_dict()
            # print(state_dict)
            # exit()
            #which has weight decay term 
            target_ids = [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 
                        122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 
                        142, 143, 144, 145, 146, 147, 148, 149, 150, 151]

            for param_id in target_ids:
                csv_file = f"./optimizer_state_report_/{param_id}.csv"

                # 파일이 비어 있을 경우 헤더 작성
                if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
                    with open(csv_file, mode="a", newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["epoch", "iteration", "exp_avg", "exp_avg_sq", "adamw_update"])

                # 파라미터가 state_dict에 있는 경우
                if param_id in state_dict['state']:
                    state = state_dict['state'][param_id]
                    if 'exp_avg' in state and 'exp_avg_sq' in state:
                        exp_avg = state['exp_avg']
                        exp_avg_sq = state['exp_avg_sq']
                        step = state['step'].item()
                        adamw_update = compute_adamw_update(exp_avg, exp_avg_sq, step, optimizer.param_groups[1]["lr"],
                                                            optimizer.param_groups[1]["betas"], optimizer.param_groups[1]["eps"])
                        exp_avg = exp_avg.abs().mean().item()
                        exp_avg_sq = exp_avg_sq.abs().mean().item()
                        with open(csv_file, mode="a", newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([epoch, iteration, exp_avg, exp_avg_sq, adamw_update])

        # # TODO: params.grad 출력 코드 
        if (device_id == 0) and (iteration % 10000 ==0):
            for name, param in model.named_parameters():
                if param.grad is not None:  #
                    if ("blocks" in name and "weight" in name):
                        if 'norm1' not in name and 'norm2' not in name and 'q_norm' not in name and 'k_norm' not in name:
                            match_block = re.search(r"modules_list\.(\d+)\.(.*)", name)
                            if match_block:
                                block_number = int(match_block.group(1))
                                layer_string = match_block.group(2)
                                probe(param.grad, block_number, layer_string, epoch, iteration)

                    elif ("head.weight" in name): 
                        # grad_array = param.grad.cpu().numpy()  
                        # filename = f"gradients_head_epoch{epoch}_iter{iteration}.npy"
                        # directory = "gradients_for_plot"  
                        # if not os.path.exists(directory):  
                        #     os.makedirs(directory)
                        # filename = os.path.join(directory, f"gradients_head_epoch{epoch}_iter{iteration}.npy")
                        # np.save(filename, grad_array)
                        # print(f"Saved gradient to {filename}")
                        probe(param.grad, 11111, 'head', epoch, iteration)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if wandb_log: 
            wandb.log({
                "training_loss": loss_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "iteration": iteration + epoch * len(data_loader)
            })
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if device_id == 0: 
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, wandb_log):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.amp.autocast("cuda"):
        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    if wandb_log:
        wandb.log({"Acc@1": metric_logger.acc1.global_avg})  # global_avg 값 사용
        wandb.log({"Acc@5": metric_logger.acc5.global_avg})  # global_avg 값 사용
        wandb.log({"val_loss/epoch": metric_logger.loss.global_avg})

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
