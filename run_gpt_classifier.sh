#!/usr/bin/env bash
set -eu

OUTPUT_DIR=./gpt_classifier_output
LOG_DIR=./logs
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

LOG="$LOG_DIR/gpt_classifier_$(date +%Y%m%d_%H%M%S).log"

echo "Starting GPT-5-mini classifier (batch mode)..."
echo "Log: $LOG"

python classification_by_prompting.py \
  --data_dir      . \
  --output_dir    "$OUTPUT_DIR" \
  --model         gpt-5-mini \
  --mode          batch \
  --val_split     0.2 \
  --seed          42 \
  --poll_interval 30 \
  2>&1 | tee "$LOG"
