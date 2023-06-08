#!/usr/bin/env bash

# CLI args with defaults
BATCH_SIZE=${1:-5}
AMP=${2:-true}
NUM_EPOCHS=${3:-30}
LEARNING_RATE=${4:-2e-5}
WEIGHT_DECAY=${5:-0.1}

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
  trip.runtime.finetune \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --gamma 0.9 \
  --cutoff 4.6 \
  --weight_decay "$WEIGHT_DECAY" \
  --data_file /results/vasp.h5 \
  --load_ckpt_path /results/model_ani1x.pth \
  --save_ckpt_path /results/model_aimd.pth \
  --seed 42 \
  --num_workers 4 \
  --gradient_clip 10.0 \
  --wandb \
  --eval_interval 1 \
  --force_weight 0.1 \
