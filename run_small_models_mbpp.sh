#!/usr/bin/env bash
set -eu

mkdir -p ./results/small_models ./logs

python main_inference.py \
    --modelNames \
        "codellama/CodeLlama-7b-Instruct-hf" \
        "deepseek-ai/deepseek-coder-6.7b-instruct" \
        "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --inputFiles \
        "./datasets/humanEval/HumanEval.jsonl" \
        "./datasets/mbpp/mbpp.jsonl" \
        "./mutations/mbpp_US_with_tests.jsonl" \
        "./mutations/mbpp_LV_with_tests.jsonl" \
        "./mutations/mbpp_SF_with_tests.jsonl" \
    --outputDir   "./results/small_models" \
    --maxNewTokens 512 \
    --timeout      50 \
    --limit        1000 \
    --gpus         0 \
    --seed         42 \
    2>&1 | tee ./logs/small_models_mbpp.log
