#!/bin/bash
LOG=./full_output/run_$(date +%Y%m%d_%H%M%S).log
mkdir -p ./full_output

CUDA_VISIBLE_DEVICES=0 python train_full_classifier.py \
  --data_dir . \
  --output_dir ./full_output \
  --model_name Qwen/Qwen2.5-Coder-1.5B \
  --max_length 512 \
  --batch_size 4 \
  --epochs 20 \
  --patience 5 \
  --lr 2e-5 \
  --dropout 0.1 \
  --val_split 0.2 \
  --seed 42 \
  --tsne \
  2>&1 | tee "$LOG"
