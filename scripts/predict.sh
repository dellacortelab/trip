#!/usr/bin/env bash

# CLI args with defaults
BATCH_SIZE=${1:-60}
AMP=${2:-true}


python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
  trip.runtime.inference \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --cutoff 4.6 \
  --use_layer_norm \
  --norm \
  --load_ckpt_path model_ani1x.pth \
  --num_workers 4 \
