#!/usr/bin/env python3
"""Merge raw batch output JSONL files with dataset records into eval-ready JSON files."""
import json, re, textwrap
from pathlib import Path

def extract_code(txt):
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
    code = m.group(1) if m else txt
    return textwrap.dedent(code).strip()

JOBS = [
    (
        "./results/api/batch_69aa0a145adc819094b2fee37beb1a16_output.jsonl",
        "./datasets/mbpp/mbpp.jsonl",
        "./results/api/gpt-5-mini__mbpp.json",
    ),
    (
        "./results/api/batch_69aa0a1406e881908b65f1b8ceafe693_output.jsonl",
        "./mutations/mbpp_US_with_tests.jsonl",
        "./results/api/gpt-5-mini__mbpp_US_with_tests.json",
    ),
    (
        "./results/api/batch_69aa0a1407748190a9a709efebcea700_output.jsonl",
        "./mutations/mbpp_SF_with_tests.jsonl",
        "./results/api/gpt-5-mini__mbpp_SF_with_tests.json",
    ),
]

for batch_file, dataset_file, output_file in JOBS:
    # Load batch responses
    responses = {}
    with open(batch_file) as f:
        for line in f:
            entry = json.loads(line)
            cid = entry["custom_id"]
            try:
                content = entry["response"]["body"]["choices"][0]["message"]["content"] or ""
            except Exception:
                content = ""
            responses[cid] = content

    # Load dataset records
    records = []
    with open(dataset_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = records[:1000]

    # Merge
    for idx, row in enumerate(records):
        response = responses.get(str(idx), "")
        code = extract_code(response)
        row_prompt = (
            row.get("mutated_prompt") or row.get("prompt") or
            row.get("original_prompt") or row.get("text") or ""
        )
        row.update({
            "GeneratedCode":     code,
            "GeneratedResponse": response,
            "PromptUsed":        row_prompt,
            "Eval_Status":       "pending",
        })

    Path(output_file).write_text(json.dumps(records, indent=2), "utf-8")
    n_with_response = sum(1 for r in records if r.get("GeneratedResponse"))
    print(f"Saved {len(records)} records ({n_with_response} with responses) → {output_file}")
