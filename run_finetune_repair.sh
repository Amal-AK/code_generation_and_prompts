#!/usr/bin/env bash
# run_finetune_repair.sh
# LoRA fine-tune a prompt repairer: mutated_prompt → original_prompt

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
MUTATION_TYPES="${2:-all}"   # all | LV | LV,SF | etc.
GPUS="${3:-0,1}"

mkdir -p finetune_repair_output repair_logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="repair_logs/run_${TIMESTAMP}.log"

echo "Model         : $MODEL"
echo "Mutations     : $MUTATION_TYPES"
echo "GPUs          : $GPUS"
echo "Log           : $LOG_FILE"
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" python finetune_prompt_repair.py \
    --modelName     "$MODEL" \
    --outputDir     finetune_repair_output \
    --mutationTypes "$MUTATION_TYPES" \
    --epochs        5 \
    --batchSize     4 \
    --gradAccum     4 \
    --lr            5e-5 \
    --warmupSteps   100 \
    --patience      3 \
    2>&1 | tee "$LOG_FILE"
