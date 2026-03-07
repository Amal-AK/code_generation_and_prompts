#!/usr/bin/env python3
"""
compute_metrics.py

Computes Pass@1, Exec%, and FP% (Partial Pass Rate) for every result JSON file
and outputs a consolidated CSV and per-dataset summary tables.

Metrics:
  Pass@1  - fraction of problems where ALL tests pass
  Exec%   - fraction of problems where code runs without crashing (Eval_Status == "OK")
  FP%     - mean(Tests_Passed / n_Tests) across all problems (partial credit)

Usage:
    python compute_metrics.py
    python compute_metrics.py --dirs results/claude results/api
    python compute_metrics.py --out my_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ── dataset / mutation detection ──────────────────────────────────────────────

DATASET_PATTERNS = {
    "HumanEval": re.compile(r"humaneval|humanEval|HumanEval", re.I),
    "MBPP":      re.compile(r"mbpp", re.I),
    "LCB":       re.compile(r"livecodebench", re.I),
}

MUTATION_PATTERNS = {
    "US": re.compile(r"_US_"),
    "LV": re.compile(r"_[Ll][Vv]_|_lv_"),
    "SF": re.compile(r"_SF_"),
}


def detect_dataset(stem: str) -> str:
    for name, pat in DATASET_PATTERNS.items():
        if pat.search(stem):
            return name
    return "Unknown"


def detect_mutation(stem: str) -> str:
    for name, pat in MUTATION_PATTERNS.items():
        if pat.search(stem):
            return name
    return "Orig"


def model_name(stem: str) -> str:
    """Extract model slug: everything before the first __ ."""
    return stem.split("__")[0]


# ── per-file metrics ───────────────────────────────────────────────────────────

def compute_file_metrics(path: Path) -> Dict:
    records = json.loads(path.read_text("utf-8"))
    n = len(records)
    if n == 0:
        return {}

    pass1   = sum(1 for r in records if r.get("Pass@1") is True)
    exec_ok = sum(1 for r in records if r.get("Eval_Status") == "OK")
    fp      = exec_ok - pass1   # runs but fails tests

    stem = path.stem
    return {
        "Model":    model_name(stem),
        "Dataset":  detect_dataset(stem),
        "Mutation": detect_mutation(stem),
        "Samples":  n,
        "Pass@1":   round(pass1 / n * 100, 1),
        "Exec%":    round(exec_ok / n * 100, 1),
        "FP%":      round(fp / n * 100, 1),
        "File":     str(path),
    }


# ── main ──────────────────────────────────────────────────────────────────────

DEFAULT_DIRS = [
    "results/small_models",
    "results/large_models",
    "results/api",
    "results/claude",
]


def main():
    parser = argparse.ArgumentParser(description="Compute Pass@1, Exec%, FP% for all result JSONs")
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                        help="directories to scan for result JSON files")
    parser.add_argument("--out", default="results/all_metrics.csv",
                        help="output CSV path")
    args = parser.parse_args()

    rows = []
    for d in args.dirs:
        for path in sorted(Path(d).glob("*.json")):
            metrics = compute_file_metrics(path)
            if metrics:
                rows.append(metrics)

    if not rows:
        print("No result files found.")
        return

    df = pd.DataFrame(rows).sort_values(["Dataset", "Model", "Mutation"])
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows → {args.out}\n")

    # ── per-dataset pivot tables ───────────────────────────────────────────────
    mutation_order = ["Orig", "US", "LV", "SF"]

    for dataset in ["HumanEval", "MBPP", "LCB"]:
        sub = df[df["Dataset"] == dataset].copy()
        if sub.empty:
            continue

        print(f"{'='*70}")
        print(f"  {dataset}")
        print(f"{'='*70}")

        # build wide table: one row per model, columns = Mutation × Metric
        pivot_rows = []
        for model, grp in sub.groupby("Model"):
            row = {"Model": model}
            for mut in mutation_order:
                m = grp[grp["Mutation"] == mut]
                if m.empty:
                    row[f"{mut}_Pass@1"] = "--"
                    row[f"{mut}_Exec%"]  = "--"
                    row[f"{mut}_FP%"]    = "--"
                else:
                    r = m.iloc[0]
                    row[f"{mut}_Pass@1"] = f"{r['Pass@1']}"
                    row[f"{mut}_Exec%"]  = f"{r['Exec%']}"
                    row[f"{mut}_FP%"]    = f"{r['FP%']}"
            pivot_rows.append(row)

        pivot = pd.DataFrame(pivot_rows)

        # multi-level header
        header1 = f"{'Model':<45}"
        header2 = f"{'Model':<45}"
        for mut in mutation_order:
            header1 += f"  {mut:^26}"
            header2 += f"  {'Pass@1':>7} {'Exec%':>7} {'FP%':>7}  "
        print(header1)
        print(header2)
        print("-" * (45 + 30 * len(mutation_order)))

        for _, row in pivot.iterrows():
            line = f"{row['Model']:<45}"
            for mut in mutation_order:
                p  = str(row.get(f"{mut}_Pass@1", "--")).rjust(7)
                e  = str(row.get(f"{mut}_Exec%",  "--")).rjust(7)
                pr = str(row.get(f"{mut}_FP%",    "--")).rjust(7)
                line += f"  {p} {e} {pr}  "
            print(line)
        print()

    print(f"Full CSV saved to: {args.out}")


if __name__ == "__main__":
    main()
