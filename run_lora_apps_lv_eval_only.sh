#!/usr/bin/env bash
# Eval-only: load saved best_lora_sft adapter from finetune_deepseek_lv_apps_output
# and run pass@1 evaluation on HumanEval LV v1 (held-out set).
# Same training configs as run_lora_apps_lv.sh — just adds --evalOnly and targets gpu3.

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./finetune_deepseek_lv_apps_output"
LOG_FILE="./logs/lora_apps_lv_evalonly_${TIMESTAMP}.log"
mkdir -p logs

CUDA_VISIBLE_DEVICES=3 python3 finetune_lora.py \
    --modelName   deepseek-ai/deepseek-coder-6.7b-instruct \
    --outputDir   "$OUTPUT_DIR" \
    --mutationTypes LV \
    --dataVariant   apps_combined \
    --evalDataset   he_v1 \
    --epochs        15 \
    --batchSize     2 \
    --gradAccum     8 \
    --lr            2e-4 \
    --maxLength     1536 \
    --loraR         16 \
    --loraAlpha     32 \
    --patience      3 \
    --warmupSteps   100 \
    --pass1EvalSamples 60 \
    --pass1EvalFreq    1 \
    --evalOnly \
    2>&1 | tee "$LOG_FILE"
