#!/bin/bash
LOG=./lora_output/run_$(date +%Y%m%d_%H%M%S).log
mkdir -p ./lora_output

CUDA_VISIBLE_DEVICES=0 python train_lora_classifier.py \
  --data_dir . \
  --output_dir ./lora_output \
  --model_name Qwen/Qwen2.5-Coder-1.5B \
  --mode lora \
  --max_length 512 \
  --batch_size 8 \
  --epochs 10 \
  --patience 5 \
  --lr 1e-4 \
  --dropout 0.1 \
  --val_split 0.2 \
  --seed 42 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --tsne \
  2>&1 | tee "$LOG"
