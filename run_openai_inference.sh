#!/usr/bin/env bash
set -eu

mkdir -p ./results/agent_fixed ./eval_agent_output ./logs

# ── Fixed prompts (HumanEval US, agent-recovered) ────────────────────────────
echo "Running gpt-5-mini on fixed prompts..."
python openai_api_inference.py \
    --model "gpt-5-mini" \
    --inputFiles "./eval_agent_output/HumanEval_US_fixed_prompts.jsonl" \
    --outputDir "./results/agent_fixed" --maxTokens 2048 --timeout 50 --no-batch \
    > ./logs/api_humaneval_US_fixed_prompts.log 2>&1

# ── GPT agent on HumanEval original ──────────────────────────────────────────
echo "Running GPT agent on HumanEval original..."
python eval_agent_only.py \
    --mutationFile "./datasets/humanEval/HumanEval.jsonl" \
    --kind         humaneval \
    --outputDir    "./eval_agent_output" \
    --gpt4Model    "gpt-4o" \
    2>&1 | tee ./logs/agent_humaneval_orig.log
