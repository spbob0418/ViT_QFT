#!/bin/bash

export OMP_NUM_THREADS=1

DIR=/home/shkim/QT/deit/output/finetune
VERSION=test
export DIR VERSION 
mkdir -p ${DIR}/${VERSION}
cd $DIR/$VERSION 

WANDB=False

export WANDB

CUDA_VISIBLE_DEVICES=4,5,6,7 taskset -c 32-63 torchrun --nproc_per_node=4 \
--master_port=$(python3 -c "import random; print(random.randint(29500, 29999))") \
/home/shkim/QT/deit/finetune.py \
--model deit_base_patch16_224 \
--sample-data-path /home/shkim/ViT_zip/sampled_imagenet/one_sample_per_class \
--register-num 4 \
--finetune True \
--transfer-learning True \
--input-size 224 \
--sched-on-updates True \
--training-steps 80000 \
--save-cp-every 100000 \
--batch-size 128 \
--accum-steps 4 \
--opt 'adamw' \
--momentum 0.9 \
--sched 'cosine' \
--weight-decay 0.01 \
--clip-grad 1 \
--data-path /data/ILSVRC2012 \
--lr 0.001 \
--output_dir ${DIR}/${VERSION} \
--warmup-epochs 1 \
--warmup-lr 1e-4 \
--min-lr 1e-5 \
--no-model-ema \
--no-repeated-aug \
--distributed > ${DIR}/${VERSION}/output.log 2>&1 &

