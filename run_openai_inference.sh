#!/usr/bin/env bash
set -eu

mkdir -p ./results/agent_fixed ./logs

# ── Fixed prompts (HumanEval US, agent-recovered) ────────────────────────────
echo "Running gpt-5-mini on fixed prompts..."
python openai_api_inference.py \
    --model "gpt-5-mini" \
    --inputFiles "./eval_agent_output/HumanEval_US_fixed_prompts.jsonl" \
    --outputDir "./results/agent_fixed" --maxTokens 2048 --timeout 50 --no-batch \
    > ./logs/api_humaneval_US_fixed_prompts.log 2>&1
