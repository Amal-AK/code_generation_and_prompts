#!/usr/bin/env python3
"""
download_apps_easy.py
─────────────────────
Download APPS introductory (easy) test problems and save as JSONL.

Output fields per record:
  task_id       : "APPS/<problem_id>"
  prompt        : problem statement (question)
  solutions     : list of canonical solution strings
  test          : LCB-format test cases [{input, output, testtype}]
  difficulty    : "introductory"

Only records with at least one solution AND at least one test case are kept.

Usage:
  python download_apps_easy.py --outputFile datasets/apps/apps_easy.jsonl
"""
import argparse
import json
import sys
from pathlib import Path


def convert_tests(input_output_raw) -> list:
    """Convert APPS {inputs:[...], outputs:[...]} → LCB [{input, output, testtype}]."""
    if not input_output_raw:
        return []
    try:
        io = (
            json.loads(input_output_raw)
            if isinstance(input_output_raw, str)
            else input_output_raw
        )
    except Exception:
        return []

    inputs  = io.get("inputs",  []) or []
    outputs = io.get("outputs", []) or []
    cases = []
    for inp, out in zip(inputs, outputs):
        cases.append({
            "input":    str(inp),
            "output":   str(out),
            "testtype": "stdin",
        })
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputFile", default="datasets/apps/apps_easy.jsonl")
    parser.add_argument("--split",      default="test",
                        help="HF split to use: train or test (default: test)")
    args = parser.parse_args()

    out_path = Path(args.outputFile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading APPS {args.split} split from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package not installed. Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset("codeparrot/apps", split=args.split, trust_remote_code=True)

    kept = skipped_no_sol = skipped_no_test = 0

    with open(out_path, "w") as fout:
        for r in ds:
            if r["difficulty"] != "introductory":
                continue

            # Parse solutions
            try:
                solutions = (
                    json.loads(r["solutions"])
                    if isinstance(r["solutions"], str)
                    else (r["solutions"] or [])
                )
            except Exception:
                solutions = []

            solutions = [s for s in solutions if s and s.strip()]
            if not solutions:
                skipped_no_sol += 1
                continue

            # Convert test cases
            test_cases = convert_tests(r.get("input_output"))
            if not test_cases:
                skipped_no_test += 1
                continue

            record = {
                "task_id":    f"APPS/{r['problem_id']}",
                "prompt":     r["question"],
                "solutions":  solutions,
                "test":       test_cases,
                "difficulty": "introductory",
            }
            fout.write(json.dumps(record) + "\n")
            kept += 1

    print(f"Saved {kept} records → {out_path}")
    print(f"Skipped: {skipped_no_sol} (no solutions), {skipped_no_test} (no tests)")


if __name__ == "__main__":
    main()
