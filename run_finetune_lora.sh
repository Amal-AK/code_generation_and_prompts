#!/usr/bin/env bash
set -eu

COMMON_ARGS="
  --dataDir       .
  --epochs        15
  --batchSize     1
  --gradAccum     16
  --lr            2e-4
  --warmupSteps   100
  --maxLength     1024
  --valSplit      0.1
  --loraR         16
  --loraAlpha     32
  --loraDropout   0.05
  --patience      4
  --pass1EvalSamples 60
  --pass1EvalFreq    1
  --evalDataset   he_v2
  --gpus          0
  --seed          42
"

# ── LV adapter — DeepSeek-Coder-6.7B-Instruct
#    Train: HumanEval LV v1 + MBPP LV v1 + MBPP LV v2
#    Eval:  HumanEval LV v2 (held-out)
mkdir -p ./finetune_deepseek_lv_output
LOG_LV=./finetune_deepseek_lv_output/run_$(date +%Y%m%d_%H%M%S).log
echo "=== Training LV adapter — DeepSeek-Coder-6.7B-Instruct ===" | tee "$LOG_LV"
python finetune_lora.py \
  $COMMON_ARGS \
  --modelName     deepseek-ai/deepseek-coder-6.7b-instruct \
  --mutationTypes LV \
  --dataVariant   mbpp_combined \
  --outputDir     ./finetune_deepseek_lv_output \
  2>&1 | tee -a "$LOG_LV"

echo "=== Done ==="
