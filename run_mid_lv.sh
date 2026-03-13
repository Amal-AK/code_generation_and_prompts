#!/usr/bin/env bash
# run_mid_lv.sh
# Multi-Interpretation Decoding on LV-mutated HumanEval.
# Usage: bash run_mid_lv.sh [MODEL] [MAX_SAMPLES] [MAX_TOKENS]
# To run all remaining models: bash run_mid_lv.sh --all

set -euo pipefail

REMAINING_MODELS=(
    "bigcode/starcoder2-15b-instruct-v0.1"
    "codellama/CodeLlama-34b-Instruct-hf"
    "deepseek-ai/deepseek-coder-33b-instruct"
    "mistralai/Codestral-22B-v0.1"
    "Qwen/Qwen2.5-Coder-32B-Instruct"
)

run_model() {
    local MODEL="$1"
    local MAX_SAMPLES="$2"
    local MAX_TOKENS="$3"

    mkdir -p MID_lv_output inference_results/mid

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
    OUTPUT_FILE="inference_results/mid/${SAFE_MODEL}__humanEval_lv_mid${MAX_SAMPLES}.jsonl"
    LOG_FILE="MID_lv_output/run_${SAFE_MODEL}_${TIMESTAMP}.log"

    echo "Model          : $MODEL"
    echo "Interpretations: $MAX_SAMPLES"
    echo "Max new tokens : $MAX_TOKENS"
    echo "Output         : $OUTPUT_FILE"
    echo "Log            : $LOG_FILE"
    echo ""

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    python multi_interpretation_decoding.py \
        --modelName    "$MODEL" \
        --inputFile    mutations/humanEval_lv_with_tests.jsonl \
        --outputFile   "$OUTPUT_FILE" \
        --maxSamples   "$MAX_SAMPLES" \
        --maxNewTokens "$MAX_TOKENS" \
        2>&1 | tee "$LOG_FILE"
}

if [[ "${1:-}" == "--all" ]]; then
    MAX_SAMPLES="${2:-5}"
    MAX_TOKENS="${3:-3000}"
    for MODEL in "${REMAINING_MODELS[@]}"; do
        echo "=========================================="
        echo "Starting: $MODEL"
        echo "=========================================="
        run_model "$MODEL" "$MAX_SAMPLES" "$MAX_TOKENS"
        echo "Finished: $MODEL"
    done
else
    MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
    MAX_SAMPLES="${2:-5}"
    MAX_TOKENS="${3:-3000}"
    run_model "$MODEL" "$MAX_SAMPLES" "$MAX_TOKENS"
fi
