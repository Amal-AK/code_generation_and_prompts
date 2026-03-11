import json
import pandas as pd
from datasets import load_dataset

OUTPUT_JSONL = "livecodebench.jsonl"
OUTPUT_EXCEL = "livecodebench.xlsx"

print("Loading LiveCodeBench dataset...")
ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True, verification_mode="no_checks")

print(f"Total problems: {len(ds)}")
print(f"Columns: {ds.column_names}")

records = [dict(row) for row in ds]

# Write JSONL
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
print(f"Saved {len(records)} records to {OUTPUT_JSONL}")

# Write Excel
df = pd.DataFrame(records)
# Stringify any list/dict columns so Excel can handle them
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False, default=str) if isinstance(x, (list, dict)) else x)
df.to_excel(OUTPUT_EXCEL, index=False)
print(f"Saved {len(records)} records to {OUTPUT_EXCEL}")
