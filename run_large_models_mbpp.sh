#!/usr/bin/env bash
set -eu

mkdir -p ./results/large_models ./logs

# CodeLlama-34b: mbpp baseline done, missing mbpp_US/LV/SF
CUDA_VISIBLE_DEVICES=1,3 python main_inference.py \
    --modelNames \
        "codellama/CodeLlama-34b-Instruct-hf" \
    --inputFiles \
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

# Qwen-32B: HumanEval baseline and all MBPP missing
CUDA_VISIBLE_DEVICES=1,3 python main_inference.py \
    --modelNames \
        "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --inputFiles \
        "./datasets/humanEval/HumanEval.jsonl" \
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
    2>&1 | tee -a ./logs/large_models_mbpp.log
