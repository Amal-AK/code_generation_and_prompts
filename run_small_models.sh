#!/usr/bin/env bash
set -eu

# Qwen-7B is missing mbpp_LV and mbpp_SF — run those only
mkdir -p ./results/small_models ./logs

python main_inference.py \
    --modelNames \
        "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --inputFiles \
        "./mutations/mbpp_LV_with_tests.jsonl" \
        "./mutations/mbpp_SF_with_tests.jsonl" \
    --outputDir   "./results/small_models" \
    --maxNewTokens 512 \
    --timeout      50 \
    --limit        1000 \
    --gpus         0 \
    --seed         42 \
    2>&1 | tee ./logs/small_models_mbpp.log
