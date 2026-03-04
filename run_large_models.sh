#!/usr/bin/env bash
set -eu

mkdir -p ./results/large_models ./logs

CUDA_VISIBLE_DEVICES=1,2 python main_inference.py \
    --modelNames \
        "bigcode/starcoder2-15b-instruct-v0.1" \
    --inputFiles \
        "./datasets/humanEval/HumanEval.jsonl" \
        "./mutations/HumanEval_US_with_tests.jsonl" \
        "./mutations/humanEval_lv_with_tests.jsonl" \
        "./mutations/humanEval_SF_with_tests.jsonl" \
        "./datasets/mbpp/mbpp.jsonl" \
        "./mutations/mbpp_US_with_tests.jsonl" \
        "./mutations/mbpp_LV_with_tests.jsonl" \
        "./mutations/mbpp_SF_with_tests.jsonl" \
    --outputDir    "./results/large_models" \
    --dtype        bfloat16 \
    --maxNewTokens 2048 \
    --timeout      50 \
    --limit        1000 \
    --seed         42 \
    2>&1 | tee ./logs/large_models.log
