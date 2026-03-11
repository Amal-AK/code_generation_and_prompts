#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="./eval_agent_output/claude_fixed_prompt_${TIMESTAMP}.log"
INPUT="./eval_agent_output/HumanEval_US_fixed_prompts.jsonl"
OUT_DIR="./results/agent_fixed"

mkdir -p "$OUT_DIR" "./eval_agent_output"

echo "Claude Sonnet 4 fixed-prompt eval — $(date)" | tee "$LOG"
echo "Input : $INPUT"                               | tee -a "$LOG"
echo "Output: $OUT_DIR"                             | tee -a "$LOG"
echo "────────────────────────────────────"         | tee -a "$LOG"

python claude_inference.py \
    --model     "claude-sonnet-4-20250514" \
    --inputFiles "$INPUT" \
    --outputDir  "$OUT_DIR" \
    --maxTokens  2048 \
    --timeout    50 \
    2>&1 | tee -a "$LOG"

echo "Done — $(date)" | tee -a "$LOG"
