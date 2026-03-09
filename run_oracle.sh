#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./oracle_output"
LOG="${OUT_DIR}/run_${TIMESTAMP}.log"
mkdir -p "$OUT_DIR"

echo "Oracle example_guided — $(date)" | tee "$LOG"

for MODEL in \
    "Qwen/Qwen2.5-Coder-32B-Instruct" \
    "deepseek-ai/deepseek-coder-33b-instruct" \
    "mistralai/Codestral-22B-v0.1" \
    "codellama/CodeLlama-34b-Instruct-hf" \
    "bigcode/starcoder2-15b-instruct-v0.1"
do
    SAFE=$(echo "$MODEL" | tr '/' '_')
    MODEL_OUT="${OUT_DIR}/${SAFE}"
    mkdir -p "$MODEL_OUT"
    echo "=== $MODEL ===" | tee -a "$LOG"

    CUDA_VISIBLE_DEVICES=1,2,3 python oracle_inference.py \
        --inputFiles  ./mutations/HumanEval_US_with_tests.jsonl \
        --modelName   "$MODEL" \
        --conditions  example_guided \
        --outputDir   "$MODEL_OUT" \
        --timeout     50 \
        2>&1 | tee -a "$LOG"

    echo "Done $MODEL — $(date)" | tee -a "$LOG"
done

echo "All done — $(date)" | tee -a "$LOG"
