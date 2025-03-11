#!/bin/bash

export OMP_NUM_THREADS=1

DIR=output
VERSION=test
export DIR VERSION 
mkdir -p ${DIR}/${VERSION}
cd $DIR/$VERSION 
WANDB=True

#fourbits_deit_small_patch16_224 --> for quantized version 
#deit_small_patch16_224 --> for fullprecision
#main_original_constant_lr
#main_original

CUDA_VISIBLE_DEVICES=4,5,6,7 taskset -c 32-63 torchrun --nproc_per_node=4 ./train_fp32.py \
--model deit_base_patch16_224 \
--epochs 450 \
--weight-decay 0.05 \
--batch-size 256 \
--repeated-aug \
--data-path /data/ILSVRC2012 \
--lr 5e-4 \
--output_dir ${DIR}/${VERSION} \
--distributed > ${DIR}/${VERSION}/output.log 2>&1 &

