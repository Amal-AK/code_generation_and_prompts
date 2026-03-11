#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./results/agent_fixed"
LOG="./eval_agent_output/fixed_prompt_eval_large_${TIMESTAMP}.log"
INPUT="./eval_agent_output/HumanEval_US_fixed_prompts.jsonl"

mkdir -p "$OUT_DIR" "./eval_agent_output"

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Fixed-prompt eval — Large models — $(date)" | tee "$LOG"
echo "Input : $INPUT"                             | tee -a "$LOG"
echo "Output: $OUT_DIR"                           | tee -a "$LOG"
echo "GPUs  : $CUDA_VISIBLE_DEVICES"              | tee -a "$LOG"
echo "────────────────────────────────────────"   | tee -a "$LOG"

python main_inference.py \
    --modelNames \
        "Qwen/Qwen2.5-Coder-32B-Instruct" \
        "deepseek-ai/deepseek-coder-33b-instruct" \
        "codellama/CodeLlama-34b-Instruct-hf" \
        "mistralai/Codestral-22B-v0.1" \
        "bigcode/starcoder2-15b-instruct-v0.1" \
    --inputFiles   "$INPUT" \
    --outputDir    "$OUT_DIR" \
    --dtype        bfloat16 \
    --maxNewTokens 512 \
    --timeout      20 \
    2>&1 | tee -a "$LOG"

echo "Done — $(date)" | tee -a "$LOG"
