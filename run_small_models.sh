#!/usr/bin/env bash
set -eu

mkdir -p ./results/small_models ./logs

# Run all small models on LiveCodeBench (base + mutations)
CUDA_VISIBLE_DEVICES=3 python main_inference.py \
    --modelNames \
        "Qwen/Qwen2.5-Coder-7B-Instruct" \
        "codellama/CodeLlama-7b-Instruct-hf" \
        "deepseek-ai/deepseek-coder-6.7b-instruct" \
    --inputFiles \
        "./datasets/livecodebench/livecodebench_public.jsonl" \
        "./mutations/livecodebench_US_with_tests.jsonl" \
        "./mutations/livecodebench_LV_with_tests.jsonl" \
        "./mutations/livecodebench_SF_with_tests.jsonl" \
    --outputDir   "./results/small_models" \
    --maxNewTokens 2048 \
    --timeout      50 \
    --limit        1000 \
    --seed         42 \
    2>&1 | tee ./logs/small_models_livecodebench.log
