"""
prepare_fixed_prompts.py
─────────────────────────
Build a JSONL input file from agent-recovered fixed prompts.
Replaces mutated_prompt with the agent's fixed_prompt so that
main_inference.py treats it as the prompt to solve.

Usage:
    python prepare_fixed_prompts.py
Outputs:
    eval_agent_output/HumanEval_US_fixed_prompts.jsonl
"""
import json, os, re

AGENT_RESULTS  = "eval_agent_output/HumanEval_US_with_tests_agent.jsonl"
MUTATION_FILE  = "mutations/HumanEval_US_with_tests.jsonl"
OUTPUT_FILE    = "eval_agent_output/HumanEval_US_fixed_prompts.jsonl"

def strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing ```python ... ``` or ``` ... ``` fences."""
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip() + "\n"

# Load agent fixed prompts
fixed = {}
with open(AGENT_RESULTS) as f:
    for line in f:
        r = json.loads(line)
        fixed[r["task_id"]] = strip_markdown_fences(r["fixed_prompt"])

# Load full mutation rows (for test, entry_point, etc.)
written = 0
os.makedirs("eval_agent_output", exist_ok=True)
with open(MUTATION_FILE) as fin, open(OUTPUT_FILE, "w") as fout:
    for line in fin:
        row = json.loads(line)
        tid = row["task_id"]
        if tid not in fixed:
            continue
        # Replace mutated_prompt with agent's fixed prompt
        row["mutated_prompt"]  = fixed[tid]
        row["original_prompt"] = fixed[tid]   # so both fields point to fixed
        fout.write(json.dumps(row) + "\n")
        written += 1

print(f"Written {written} rows → {OUTPUT_FILE}")
