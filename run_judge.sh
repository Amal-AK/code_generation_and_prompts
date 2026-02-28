LOG=./judge_output/run_lcb_$(date +%Y%m%d_%H%M%S).log
mkdir -p ./judge_output

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python mutation_judge.py \
  --gpus 0,1,2,3 \
  --judgeModel Qwen/Qwen2.5-Coder-32B-Instruct \
  --batchSize 8 \
  --inputFiles \
    ./mutations/livecodebench_LV_with_tests.jsonl \
    ./mutations/livecodebench_SF_with_tests.jsonl \
    ./mutations/livecodebench_US_with_tests.jsonl \
  --outputDir ./judge_output/full \
  2>&1 | tee "$LOG"
