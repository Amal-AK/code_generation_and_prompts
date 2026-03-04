#!/bin/bash
LOG=./finetune_lv_sf_output/run_$(date +%Y%m%d_%H%M%S).log
mkdir -p ./finetune_lv_sf_output

CUDA_VISIBLE_DEVICES=1,3 python finetune_lora.py \
  --modelName     Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataDir       . \
  --outputDir     ./finetune_lv_sf_output \
  --mutationTypes LV,SF \
  --epochs        3 \
  --batchSize     2 \
  --gradAccum     8 \
  --lr            2e-4 \
  --maxLength     1024 \
  --valSplit      0.1 \
  --loraR         16 \
  --loraAlpha     32 \
  --loraDropout   0.05 \
  --patience      3 \
  --gpus          1,3 \
  --seed          42 \
  2>&1 | tee "$LOG"
