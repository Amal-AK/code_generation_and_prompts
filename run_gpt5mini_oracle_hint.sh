#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="./eval_agent_output/gpt5mini_oracle_hint_${TIMESTAMP}.log"
INPUT="./eval_agent_output/HumanEval_US_oracle_hint_prompts.jsonl"
OUT_DIR="./results/agent_fixed"

mkdir -p "$OUT_DIR" "./eval_agent_output"

echo "gpt-5-mini oracle hint eval — $(date)" | tee "$LOG"
echo "Input : $INPUT"                        | tee -a "$LOG"
echo "Output: $OUT_DIR"                      | tee -a "$LOG"
echo "────────────────────────────────────"  | tee -a "$LOG"

python openai_api_inference.py \
    --model      gpt-5-mini \
    --inputFiles "$INPUT" \
    --outputDir  "$OUT_DIR" \
    --no-batch \
    --timeout    50 \
    2>&1 | tee -a "$LOG"

echo "Done — $(date)" | tee -a "$LOG"
