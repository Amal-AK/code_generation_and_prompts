#!/usr/bin/env bash
set -eu

mkdir -p ./results/claude ./logs

# ── LiveCodeBench original: single batch job ─────────────────────────────────
echo "Submitting LiveCodeBench original batch job..."

python claude_inference.py \
    --model "claude-sonnet-4-20250514" \
    --inputFiles "./datasets/livecodebench/livecodebench_public.jsonl" \
    --outputDir "./results/claude" --maxTokens 2048 --timeout 50 --limit 1000 \
    --skip-eval \
    2>&1 | tee ./logs/claude_lcb_orig.log

echo "Done. Responses saved to ./results/claude/"

# ── Eval LCB original response file ──────────────────────────────────────────
python claude_inference.py \
    --model     "claude-sonnet-4-20250514" \
    --outputDir "./results/claude" \
    --timeout   50 \
    --eval-only \
        "./results/claude/claude-sonnet-4-20250514__livecodebench_public.json" \
    2>&1 | tee ./logs/claude_lcb_orig_eval.log
