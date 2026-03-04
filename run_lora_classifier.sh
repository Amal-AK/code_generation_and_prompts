#!/bin/bash
LOG=./lora_1b_output/run_$(date +%Y%m%d_%H%M%S).log
mkdir -p ./lora_1b_output

CUDA_VISIBLE_DEVICES=2 python train_lora_classifier.py \
  --data_dir . \
  --output_dir ./lora_1b_output \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --gpus 2 \
  --mode lora \
  --max_length 512 \
  --batch_size 32 \
  --grad_accum 4 \
  --epochs 10 \
  --patience 5 \
  --lr 2e-4 \
  --dropout 0.1 \
  --val_split 0.2 \
  --seed 123456 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --tsne \
  2>&1 | tee "$LOG"
