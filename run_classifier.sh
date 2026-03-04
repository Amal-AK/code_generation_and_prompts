#!/bin/bash
LOG=./classifier_1b_output/run_$(date +%Y%m%d_%H%M%S).log
mkdir -p ./classifier_1b_output

CUDA_VISIBLE_DEVICES=2 python train_classifier.py \
  --data_dir . \
  --output_dir ./classifier_1b_output \
  --model_name Qwen/Qwen2.5-Coder-1.5B \
  --max_length 512 \
  --batch_size 32 \
  --epochs 20 \
  --patience 5 \
  --lr 3e-4 \
  --dropout 0.1 \
  --val_split 0.2 \
  --seed 123456 \
  --tsne \
  2>&1 | tee "$LOG"
