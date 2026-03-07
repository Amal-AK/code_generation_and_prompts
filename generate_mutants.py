#!/usr/bin/env python3
"""
generate_lv_mutants.py

Batch-mode Lexical Vagueness (LV) prompt mutation pipeline for code-generation benchmarks
(HumanEval / MBPP) using GPT-5-mini via the OpenAI Batch API.

This version includes a fix for AttributeError when inspecting batch.request_counts
by treating that object as a model and using getattr(...).
"""

import argparse
import json
import time
from typing import Dict, Any, List
from openai import OpenAI
import os
import sys

# --------------------
# Configuration
# --------------------
MODEL_NAME = "gpt-5-mini"
INPUT_FILE = "./datasets/humanEval/HumanEval.jsonl"
BATCH_INPUT_FILE = "HumanEval_LV_input.jsonl"
BATCH_OUTPUT_FILE = "HumanEval_LV_output.jsonl"
OUTPUT_FILE = "mutations/variant2/humanEval_LV_v2_mutated.jsonl"
POLL_INTERVAL = 10  # seconds
# --------------------

client = OpenAI()










SYSTEM_PROMPT = """
You are a controlled prompt-mutation agent for dataset construction.
Global rules:
- Apply ONLY the requested MUTATION type. 
- If mutation is not applicable, return applicable=false.
- Do NOT provide any chain-of-thought, reasoning, or explanations.
- Return EXACTLY one JSON object and nothing else. 
- mutated_prompt must be a direct string (no surrounding markdown), preserve examples verbatim, and be ≤ 130% length of original.

"""






USER_PROMPT_LV = """
INPUT:
{{"specific_prompt":"{prompt_text}"}}

MUTATION: Lexical Vagueness

Goal:
Make the prompt noticeably less precise by generalizing or blurring wording,
while preserving the core task goal and I/O structure.

Rules:

- Replace only clear, well-defined terms with broader ones.
- You MAY generalize concrete nouns or conditions.
- You MAY weaken action verbs.
- Apply multiple related lexical changes (not a single-word swap).
- Apply Lexical Vagueness ONLY.
- Do NOT change the task goal.
- Do not delete the imports. 
- Do NOT add new constraints, numbers, libraries, or conceptual abstractions.
- Do NOT change function signatures, parameters, examples, or I/O.
- Do NOT introduce contradictions or other mutation types.
- Rename the function and parameter identifiers by synonyms to be less informative, and use the same name for the examples. keep the order of parameters. 
- Generalize or remove type annotations but do not change the number/position of arguments.
- You MAY weaken examples slightly (e.g., replace specific numeric values with generic placeholders). 
- If the prompt is extremely short or already very generic, return applicable=false.


OUTPUT (JSON only).

If not applicable:
{{"mutation_type":"LV","applicable":false}}

If applicable:
{{"mutation_type":"LV","applicable":true,"mutated_prompt":"<text>"}}
"""


USER_PROMPT_SF = """
INPUT:
{{"specific_prompt":"{prompt_text}"}}

MUTATION: Syntax & Formatting (Robustness Stress)

Goal:
Produce a mutated variant of the given prompt that introduces realistic surface-level syntax or formatting corruption while preserving the original task semantics.

RULES:

- Introduce between 3 and 6 surface-level syntax/formatting errors. 
- Errors must belong to at least TWO different categories:
  (1) Token-level corruption (typo, swapped character)
  (2) Delimiter/bracket/colon corruption
  (3) Indentation or layout corruption
  (4) Example formatting corruption
- At least ONE error must affect structure (delimiter, colon, indentation, bracket, fence, doctest marker).
- ONLY introduce surface-level corruption. Do not change task meaning.
- Do NOT change the task goal, add libraries, numeric constants, credentials, or assumptions.
- Errors may interact locally (e.g., header + indentation), but must not make the task irrecoverable.
- The mutated prompt must remain readable. 
- Do not use the same operator all the time, samples from different categories.
- Output must be valid JSON.

OUTPUT (JSON only):
{{"mutation_type":"SF","applicable":true,"mutated_prompt":"<text>"}}
"""




USER_PROMPT_UNDERSPECIFICATION = """Remove exactly one constraint from the prompt below that changes what a correct implementation must handle.

WHAT TO REMOVE (pick one):
- Constraint bounds: "1 <= k <= 15", "n <= 10^5", "at least 4 characters"
- Ordering/tie-breaking: "sorted", "return the first one in case of ties"
- Input precondition: "starts with zero balance", "Ignore spaces", "positive", "non-empty"
- Output edge-case: "return None if empty", "Empty sum equals 0"
- Formula pinning behavior: "MAD = average |x - x_mean|"
- One-sentence prompts: MUST always mutate — remove "first"/"last"/"minimum"/"maximum"/"distinct"/"non-repeated"/"sorted"/"consecutive"/"all"/"specific"/"singleton" or whichever word most limits scope.

DO NOT: swap words for synonyms; remove examples or sample I/O; remove "using X" hints (lambda/regex/heap); remove the core concept definition; remove more than one thing; add or rewrite any text.

IMPORTANT: applicable is ALWAYS true. Never return applicable=false. If nothing else fits, remove any scope-limiting word ("all", "specific", "each", "consecutive", "singleton", "minimum", "first").

Delete the constraint completely, fix grammar if needed, keep everything else verbatim.
mutated_prompt must contain ONLY the modified prompt — no instructions or meta-text.

<original_prompt>
{prompt_text}
</original_prompt>

OUTPUT (JSON only): {{"mutation_type":"US","applicable":true,"mutated_prompt":"<modified prompt text only>"}}
"""

MUTATION_PROMPTS = {
    "LV": USER_PROMPT_LV,
    "SF": USER_PROMPT_SF,
    "US" :USER_PROMPT_UNDERSPECIFICATION, 
}

# --------------------
# Utility functions
# --------------------
def parse_json(raw: str) -> Dict[str, Any]:
    """Parse the model-returned JSON string into a dict; strip markdown fences if present."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("\n", 1)
            cleaned = parts[1] if len(parts) > 1 else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by model.\nError: {e}\nRaw output:\n{raw}")


def validate_lv_mutation(original: str, mutated: str) -> bool:
    if not mutated or not mutated.strip():
        return False

    # Normalize whitespace for robust comparison
    orig_norm = " ".join(original.split())
    mut_norm = " ".join(mutated.split())

    if orig_norm == mut_norm:
        return False

    return True



def build_batch_input_line(task_id: str, prompt_text: str, mutation_type: str = "LV") -> str:
    # Use the mutation_type parameter to select which prompt
    user_prompt_template = MUTATION_PROMPTS.get(mutation_type, USER_PROMPT_LV)
    user_prompt = user_prompt_template.format(
        task_id=task_id,
        prompt_text=prompt_text.replace('"', '\\"')
    )
    request_obj = {
        "custom_id": task_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
    }
    return json.dumps(request_obj, ensure_ascii=False)


# --------------------
# Batch workflow
# --------------------
def create_and_upload_batch_input(tasks: List[Dict[str, str]], mutation_type: str = "LV") -> str:  
    with open(BATCH_INPUT_FILE, "w", encoding="utf-8") as bf:
        for t in tasks:
            line = build_batch_input_line(t["task_id"], t["prompt"], mutation_type)  
            bf.write(line + "\n")
    print(f"Wrote {len(tasks)} lines to {BATCH_INPUT_FILE}")

    print("Uploading batch input file...")
    with open(BATCH_INPUT_FILE, "rb") as fh:
        upload = client.files.create(file=fh, purpose="batch")
    input_file_id = getattr(upload, "id", None)
    if not input_file_id:
        raise RuntimeError("Upload returned no file id")
    print("Uploaded. input_file_id =", input_file_id)
    return input_file_id


def create_batch_job(input_file_id: str) -> str:
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    batch_id = getattr(batch, "id", None)
    print("Batch created:", batch_id, "initial status:", getattr(batch, "status", None))
    if not batch_id:
        raise RuntimeError("Failed to create batch job")
    return batch_id


def poll_batch_until_done(batch_id: str, poll_interval: int = POLL_INTERVAL) -> Any:
    """
    Poll the batch job until it reaches 'completed' (or fails/expired).
    Fixed: request_counts may be a model object not a dict — use getattr to access fields.
    """
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None)
        # request_counts may be a pydantic model object with attributes, so use getattr
        rc = getattr(batch, "request_counts", None)
        if rc is None:
            completed = "?"
        else:
            # try attribute access defensively
            completed = getattr(rc, "completed", getattr(rc, "completed_requests", "?"))
        print(f"Batch {batch_id} status: {status} (completed: {completed})")
        if status == "completed":
            return batch
        if status in ("failed", "expired"):
            # include errors if present
            errors = getattr(batch, "errors", None)
            raise RuntimeError(f"Batch {batch_id} ended with status {status}. Batch errors: {errors}")
        time.sleep(poll_interval)


def download_batch_output(batch: Any) -> bytes:
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("No output_file_id on completed batch")
    print("Downloading batch output file:", output_file_id)
    file_content = client.files.content(output_file_id)
    # handle different return shapes
    if hasattr(file_content, "content"):
        raw_bytes = file_content.content
    else:
        raw_bytes = file_content
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-8")
    with open(BATCH_OUTPUT_FILE, "wb") as out_f:
        out_f.write(raw_bytes)
    print("Saved batch output to", BATCH_OUTPUT_FILE)
    return raw_bytes

def process_batch_output(tasks: List[Dict[str, str]], mutation_type: str = "LV") -> None:
    original_by_id = {t["task_id"]: t["prompt"] for t in tasks}

    with open(BATCH_OUTPUT_FILE, "r", encoding="utf-8") as in_f, \
        open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:

        for line in in_f:
            if not line.strip():
                continue

            entry = json.loads(line)
            custom_id = entry.get("custom_id")

            # Check for top-level errors
            if entry.get("error"):
                rec = {
                    "task_id": custom_id,
                    "original_prompt": original_by_id.get(custom_id, ""),
                    "mutation_type": mutation_type,
                    "applicable": False,
                    "error": str(entry.get("error"))
                }
                out_f.write(json.dumps(rec) + "\n")
                continue

            # Navigate: entry -> response -> body -> choices -> message -> content
            response_wrapper = entry.get("response")
            if not response_wrapper:
                rec = {
                    "task_id": custom_id,
                    "original_prompt": original_by_id.get(custom_id, ""),
                    "mutation_type": mutation_type,
                    "applicable": False,
                    "error": "no_response_field"
                }
                out_f.write(json.dumps(rec) + "\n")
                continue

            # Check status_code instead of status
            status_code = response_wrapper.get("status_code")
            if status_code and status_code != 200:
                rec = {
                    "task_id": custom_id,
                    "original_prompt": original_by_id.get(custom_id, ""),
                    "mutation_type": mutation_type,
                    "applicable": False,
                    "error": f"status_code_{status_code}"
                }
                out_f.write(json.dumps(rec) + "\n")
                continue

            # Extract body
            body = response_wrapper.get("body", {})
            if not body or body.get("error"):
                rec = {
                    "task_id": custom_id,
                    "original_prompt": original_by_id.get(custom_id, ""),
                    "mutation_type": mutation_type,
                    "applicable": False,
                    "error": str(body.get("error")) if body else "no_body"
                }
                out_f.write(json.dumps(rec) + "\n")
                continue

            # Extract content from choices
            content = None
            try:
                choices = body.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    content = choices[0].get("message", {}).get("content")
            except Exception:
                content = None

            if not content:
                rec = {
                    "task_id": custom_id,
                    "original_prompt": original_by_id.get(custom_id, ""),
                    "mutation_type": mutation_type,
                    "applicable": False,
                    "error": "no_content_in_message"
                }
                out_f.write(json.dumps(rec) + "\n")
                continue

            # Parse the JSON from the model
            try:
                parsed = parse_json(content)
            except Exception as e:
                rec = {
                    "task_id": custom_id,
                    "original_prompt": original_by_id.get(custom_id, ""),
                    "mutation_type": mutation_type,
                    "applicable": False,
                    "error": f"parse_error: {e}",
                    "raw_content": content if len(content) < 2000 else None
                }
                out_f.write(json.dumps(rec) + "\n")
                continue

            # Extract applicable field
            model_applicable = parsed.get("applicable")
            if isinstance(model_applicable, str):
                model_applicable = model_applicable.lower() in ("true", "1", "yes")
            else:
                model_applicable = bool(model_applicable)

            # Build output record
            if not model_applicable:
                rec = {
                    "task_id": custom_id,
                    "original_prompt": original_by_id.get(custom_id, ""),
                    "mutation_type": mutation_type,
                    "applicable": False
                }
            else:
                mutated = parsed.get("mutated_prompt") or parsed.get("ambiguous_prompt") or ""
                rec = {
                    "task_id": custom_id,
                    "original_prompt": original_by_id.get(custom_id, ""),
                    "mutation_type": mutation_type,
                    "applicable": True,
                    "mutated_prompt": mutated
                }

            out_f.write(json.dumps(rec) + "\n")

    print("Wrote final output to", OUTPUT_FILE)



def read_input_jsonl(path: str) -> List[Dict[str, Any]]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            
            task_id = obj.get("task_id") or obj.get("id")
            prompt_text = obj.get("prompt") or obj.get("text")

            if task_id is None or prompt_text is None:
                print("Skipping malformed line:", obj, file=sys.stderr)
                continue

            tasks.append({
                "task_id": str(task_id),
                "prompt": prompt_text
            })
    return tasks





def main(args):
    print("Reading input dataset:", args.input)
    tasks = read_input_jsonl(args.input)
    if not tasks:
        print("No tasks found in input file. Exiting.")
        return
    print(f"Loaded {len(tasks)} tasks. Preparing batch input...")
    input_file_id = create_and_upload_batch_input(tasks, args.mutation_type)  # ADD args.mutation_type
    batch_id = create_batch_job(input_file_id)
    print("Polling batch job until completion. This may take some time...")
    final_batch = poll_batch_until_done(batch_id, poll_interval=args.poll_interval)
    print("Downloading outputs...")
    _ = download_batch_output(final_batch)
    print("Processing batch outputs into final output file...")
    process_batch_output(tasks, args.mutation_type)
    print("Done. Lexical Vagueness mutants written to", OUTPUT_FILE)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch mutant generator (OpenAI Batch API)")
    p.add_argument("--input", type=str, default=INPUT_FILE, help="Path to input JSONL dataset (one object per line)")
    p.add_argument("--poll-interval", type=int, default=POLL_INTERVAL, help="Seconds between batch status polls")
    p.add_argument("--mutation_type", type=str, default='SF', help="which mutation type to apply: 'LV', 'SF', or 'US'")
    args = p.parse_args()
    main(args)
