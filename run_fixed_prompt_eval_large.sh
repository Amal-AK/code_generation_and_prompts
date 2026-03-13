#!/usr/bin/env bash
set -eu

mkdir -p ./inference_results/lv_fname_experiment ./logs

python main_inference.py \
    --modelNames \
        "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --inputFiles \
        "./datasets/humanEval/HumanEval.jsonl" \
        "./mutations/humanEval_lv_with_tests.jsonl" \
        "./mutations/humanEval_LV_restored_fname_with_tests.jsonl" \
    --outputDir   "./inference_results/lv_fname_experiment" \
    --maxNewTokens 512 \
    --timeout      50 \
    --limit        1000 \
    --gpus         0 \
    --seed         42 \
    2>&1 | tee ./logs/lv_fname_experiment.log
