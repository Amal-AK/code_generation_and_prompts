#!/usr/bin/env bash
# run_self_consistency_lv.sh
# Interpretation-aware self-consistency on LV-mutated HumanEval.

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
MAX_SAMPLES="${2:-10}"
MAX_TOKENS="${3:-3000}"

mkdir -p sc_lv_output results/self_consistency

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
OUTPUT_FILE="results/self_consistency/${SAFE_MODEL}__humanEval_lv_sc${MAX_SAMPLES}.jsonl"
LOG_FILE="sc_lv_output/run_${TIMESTAMP}.log"

echo "Model          : $MODEL"
echo "Interpretations: $MAX_SAMPLES"
echo "Max new tokens : $MAX_TOKENS"
echo "Output         : $OUTPUT_FILE"
echo "Log            : $LOG_FILE"
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
python self_consistency_lv.py \
    --modelName    "$MODEL" \
    --inputFile    mutations/humanEval_lv_with_tests.jsonl \
    --outputFile   "$OUTPUT_FILE" \
    --maxSamples   "$MAX_SAMPLES" \
    --maxNewTokens "$MAX_TOKENS" \
    2>&1 | tee "$LOG_FILE"
