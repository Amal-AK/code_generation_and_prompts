#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./eval_agent_output"
LOG="${OUT_DIR}/run_${TIMESTAMP}.log"
mkdir -p "$OUT_DIR"

GEN_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
GPT_MODEL="gpt-4o"

echo "US Recovery Agent Eval — $(date)" | tee "$LOG"
echo "Gen model  : $GEN_MODEL"          | tee -a "$LOG"
echo "GPT model  : $GPT_MODEL"          | tee -a "$LOG"
echo "Output     : $OUT_DIR"            | tee -a "$LOG"
echo "────────────────────────────────" | tee -a "$LOG"

python eval_agent_only.py \
    --mutationFile  mutations/HumanEval_US_with_tests.jsonl \
    --kind          humaneval \
    --genModel      "$GEN_MODEL" \
    --outputDir     "$OUT_DIR" \
    --gpt4Model     "$GPT_MODEL" \
    2>&1 | tee -a "$LOG"

echo "Done — $(date)" | tee -a "$LOG"
