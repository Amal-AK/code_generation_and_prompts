#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./oracle_output"
LOG="${OUT_DIR}/run_${TIMESTAMP}.log"
mkdir -p "$OUT_DIR"

echo "Oracle study — $(date)" | tee "$LOG"

CUDA_VISIBLE_DEVICES=0,1,2,3 python oracle_inference.py \
    --inputFiles \
        ./mutations/humanEval_lv_with_tests.jsonl \
        ./mutations/humanEval_SF_with_tests.jsonl \
        ./mutations/HumanEval_US_with_tests.jsonl \
    --modelName  Qwen/Qwen2.5-Coder-7B-Instruct \
    --conditions baseline oracle example_guided \
    --outputDir  "$OUT_DIR" \
    --timeout    50 \
    2>&1 | tee -a "$LOG"

echo "Done — $(date)" | tee -a "$LOG"
