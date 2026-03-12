#!/usr/bin/env bash
set -eu

mkdir -p ./results/large_models ./logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1,2,3 python main_inference.py \
    --modelNames \
        "deepseek-ai/deepseek-coder-33b-instruct" \
    --inputFiles \
        "./datasets/livecodebench/livecodebench_public.jsonl" \
        "./mutations/livecodebench_US_with_tests.jsonl" \
        "./mutations/livecodebench_LV_with_tests.jsonl" \
        "./mutations/livecodebench_SF_with_tests.jsonl" \
    --outputDir    "./results/large_models" \
    --dtype        bfloat16 \
    --maxNewTokens 2048 \
    --timeout      50 \
    --limit        1000 \
    --seed         42 \
    2>&1 | tee ./logs/deepseek33_lcb_rerun.log
