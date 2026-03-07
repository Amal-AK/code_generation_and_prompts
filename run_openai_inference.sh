#!/usr/bin/env bash
set -eu

mkdir -p ./results/api ./logs

# ── LCB: submit all 4 batch jobs in parallel ─────────────────────────────────
echo "Submitting 4 LCB batch jobs in parallel..."

python openai_api_inference.py \
    --model "gpt-5-mini" \
    --inputFiles "./datasets/livecodebench/livecodebench_public.jsonl" \
    --outputDir "./results/api" --maxTokens 2048 --timeout 60 --limit 1000 \
    --skip-eval \
    > ./logs/api_lcb_orig.log 2>&1 &

python openai_api_inference.py \
    --model "gpt-5-mini" \
    --inputFiles "./mutations/livecodebench_US_with_tests.jsonl" \
    --outputDir "./results/api" --maxTokens 2048 --timeout 60 --limit 1000 \
    --skip-eval \
    > ./logs/api_lcb_us.log 2>&1 &

python openai_api_inference.py \
    --model "gpt-5-mini" \
    --inputFiles "./mutations/livecodebench_LV_with_tests.jsonl" \
    --outputDir "./results/api" --maxTokens 2048 --timeout 60 --limit 1000 \
    --skip-eval \
    > ./logs/api_lcb_lv.log 2>&1 &

python openai_api_inference.py \
    --model "gpt-5-mini" \
    --inputFiles "./mutations/livecodebench_SF_with_tests.jsonl" \
    --outputDir "./results/api" --maxTokens 2048 --timeout 60 --limit 1000 \
    --skip-eval \
    > ./logs/api_lcb_sf.log 2>&1 &

echo "All 4 LCB batch jobs submitted. Waiting for completion..."
