#!/usr/bin/env bash
set -eu

mkdir -p ./results/large_models ./logs

CUDA_VISIBLE_DEVICES=1,2,3 python main_inference.py \
    --modelNames \
        "Qwen/Qwen2.5-Coder-32B-Instruct" \
        "codellama/CodeLlama-34b-Instruct-hf" \
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
    2>&1 | tee ./logs/large_models_lcb.log
