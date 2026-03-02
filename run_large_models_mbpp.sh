#!/usr/bin/env bash
set -eu

# CodeLlama-34b and Qwen-32B already have HumanEval results — MBPP only
mkdir -p ./results/large_models ./logs

CUDA_VISIBLE_DEVICES=1,2,3 python main_inference.py \
    --modelNames \
        "codellama/CodeLlama-34b-Instruct-hf" \
        "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --inputFiles \
        "./datasets/mbpp/mbpp.jsonl" \
        "./mutations/mbpp_US_with_tests.jsonl" \
        "./mutations/mbpp_LV_with_tests.jsonl" \
        "./mutations/mbpp_SF_with_tests.jsonl" \
    --outputDir    "./results/large_models" \
    --dtype        bfloat16 \
    --maxNewTokens 512 \
    --timeout      50 \
    --limit        1000 \
    --seed         42 \
    2>&1 | tee ./logs/large_models_mbpp.log
