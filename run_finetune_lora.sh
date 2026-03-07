#!/usr/bin/env bash
set -eu

COMMON_ARGS="
  --modelName     Qwen/Qwen2.5-Coder-7B-Instruct
  --dataDir       .
  --epochs        10
  --batchSize     2
  --gradAccum     8
  --lr            2e-4
  --maxLength     1024
  --valSplit      0.1
  --loraR         16
  --loraAlpha     32
  --loraDropout   0.05
  --patience      3
  --gpus          0
  --seed          42
"

# ── LV adapter (v1) ───────────────────────────────────────────────────────────
mkdir -p ./finetune_lv_output
LOG_LV=./finetune_lv_output/run_$(date +%Y%m%d_%H%M%S).log
echo "=== Training LV adapter (v1) ===" | tee "$LOG_LV"
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
  $COMMON_ARGS \
  --mutationTypes LV \
  --dataVariant   v1 \
  --outputDir     ./finetune_lv_output \
  2>&1 | tee -a "$LOG_LV"

# ── LV adapter (v2) ───────────────────────────────────────────────────────────
mkdir -p ./finetune_lv_v2_output
LOG_LV2=./finetune_lv_v2_output/run_$(date +%Y%m%d_%H%M%S).log
echo "=== Training LV adapter (v2) ===" | tee "$LOG_LV2"
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
  $COMMON_ARGS \
  --mutationTypes LV \
  --dataVariant   v2 \
  --outputDir     ./finetune_lv_v2_output \
  2>&1 | tee -a "$LOG_LV2"

# ── SF adapter ────────────────────────────────────────────────────────────────
mkdir -p ./finetune_sf_output
LOG_SF=./finetune_sf_output/run_$(date +%Y%m%d_%H%M%S).log
echo "=== Training SF adapter ===" | tee "$LOG_SF"
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
  $COMMON_ARGS \
  --mutationTypes SF \
  --outputDir     ./finetune_sf_output \
  2>&1 | tee -a "$LOG_SF"

echo "=== All adapters done ==="
