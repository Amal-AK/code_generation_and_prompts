"""
prepare_oracle_hint_prompts.py
Inject the US example-guided oracle hint into mutated_prompt for each row,
producing a JSONL consumable by openai_api_inference.py.
"""
import json, os

HINT = (
    "\nNote: The description above is underspecified — a key constraint has been omitted. "
    "However, the input/output examples in the docstring reflect the *complete* intended behavior.\n\n"
    "Follow these steps:\n"
    "Step 1 — Examine each example carefully: trace what relationship the inputs and output must satisfy.\n"
    "Step 2 — In one sentence starting with exactly 'Missing constraint: ', state the constraint "
    "the examples reveal but the description omits.\n"
    "Step 3 — Implement the function that satisfies that constraint."
)

INPUT  = "mutations/HumanEval_US_with_tests.jsonl"
OUTPUT = "eval_agent_output/HumanEval_US_oracle_hint_prompts.jsonl"

os.makedirs("eval_agent_output", exist_ok=True)
written = 0
with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        row = json.loads(line)
        if not row.get("mutated_prompt"):
            continue
        row["mutated_prompt"] = row["mutated_prompt"].rstrip() + HINT
        fout.write(json.dumps(row) + "\n")
        written += 1

print(f"Written {written} rows → {OUTPUT}")
