# Fixed-Prompt Eval — Small Models Results
Run: 2026-03-09 (fixed_prompt_eval_20260309_172733.log)
Dataset: eval_agent_output/HumanEval_US_fixed_prompts.jsonl (n=160)
Prompts: GPT-4o restored (via recovery_agent.py)
Eval: all tests must pass (pass == n_tests)

| Model                              | Pass@1 | SuccessExec |
|------------------------------------|--------|-------------|
| gpt-5-mini                         | 0.775  | 0.931       |
| Qwen/Qwen2.5-Coder-7B-Instruct     | 0.681  | 0.975       |
| deepseek-ai/deepseek-coder-6.7b-instruct | 0.625 | 0.969  |
| codellama/CodeLlama-7b-Instruct-hf | 0.344  | 0.894       |

Large models (Qwen 32B, DeepSeek 33B, CodeLlama 34B, Codestral 22B, StarCoder2 15B):
→ Run aborted (CPU fallback). Rerun with run_fixed_prompt_eval_large.sh (3 GPUs).
