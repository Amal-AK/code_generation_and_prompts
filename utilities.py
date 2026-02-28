#!/usr/bin/env python3
"""
excel_to_shuffled_json.py

Convert a 2-column Excel workbook (columns: prompt, label)
into a shuffled JSON list.

Usage
-----
python excel_to_shuffled_json.py input.xlsx output.json
"""

import json
import random
import sys
from pathlib import Path
import openpyxl
import pandas as pd


import json

def label_prompts(file1_path, file2_path, output_path, clear_count=300):
    # Load first file and label everything 'problematic'
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    labeled = [
        {"prompt": item["prompt"], "label": "problematic"}
        for item in data1
    ]

    # Load second file and take first `clear_count` as 'clair'
    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    for item in data2[:clear_count]:
        labeled.append({"prompt": item["prompt"], "label": "clair"})

    # Write out the combined labeled list
    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(labeled, out, ensure_ascii=False, indent=2)




def excel_to_shuffled_json(xls_path: str | Path,
                           json_path: str | Path,
                           sheet: str | int | None = 0,
                           seed: int | None = 42,
                           encoding: str = "utf-8") -> None:
    xls_path, json_path = Path(xls_path), Path(json_path)

    # ── read Excel ──────────────────────────────────────────────────────
    df = pd.read_excel(xls_path, sheet_name=sheet, dtype=str)

    # keep only “prompt” and “label” columns (renaming just in case)
    df = df.rename(columns=lambda c: c.strip().lower())
    if not {"prompt", "label"} <= set(df.columns):
        raise ValueError("The sheet must contain columns named 'prompt' and 'label'.")

    data = df[["prompt", "label"]].to_dict(orient="records")

    # ── shuffle (deterministically, unless seed=None) ───────────────────
    if seed is not None:
        random.seed(seed)
    random.shuffle(data)

    # ── write JSON ──────────────────────────────────────────────────────
    with open(json_path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅  {len(data)} rows shuffled and written → {json_path}")


if __name__ == "__main__":
    

    #excel_to_shuffled_json("./datasets/HumanEval/humaneval_classify_prompts.xlsx", "./datasets/HumanEval/classify_prompts_humaneval.json", seed=42)
    label_prompts("./datasets/HumanEval/classify_prompts_humaneval.json","./datasets/HumanEval/HumanEval.json", "./datasets/HumanEval/clair_problematic_HumanEval.json", clear_count=300)