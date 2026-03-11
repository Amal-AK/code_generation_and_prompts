#!/usr/bin/env bash
# Train LoRA on MBPP v1+v2 + APPS LV, eval on HumanEval LV v1 (fully held out).
# Target format: mutated_prompt -> clarified original_prompt + solution_code

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./finetune_deepseek_lv_apps_output"
LOG_FILE="./logs/lora_apps_lv_${TIMESTAMP}.log"
mkdir -p logs

python3 finetune_lora.py \
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
    --gpus          0,1,2,3 \
    2>&1 | tee "$LOG_FILE"
