#!/usr/bin/env bash
set -eu

# deepseek-33b, Codestral-22B, starcoder2-15b — no results yet, run all inputs
mkdir -p ./results/large_models ./logs

CUDA_VISIBLE_DEVICES=0,2 python main_inference.py \
    --modelNames \
        "deepseek-ai/deepseek-coder-33b-instruct" \
        "mistralai/Codestral-22B-v0.1" \
        "bigcode/starcoder2-15b" \
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
    --maxNewTokens 512 \
    --timeout      50 \
    --limit        1000 \
    --seed         42 \
    2>&1 | tee ./logs/large_models_remaining.log
