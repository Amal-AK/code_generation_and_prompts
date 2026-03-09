"""
us_recovery_agent.py
────────────────────
GPT-4-powered test-guided constraint recovery for Under-Specification (US) mutations.

Flow per sample:

  US mutated prompt + test row
        │
        ▼
  GPT-4: "What constraint was removed? Give N hypotheses."
        │
        for each hypothesis (up to max_hypotheses):
          │
          ▼
  GPT-4: "Restore this constraint into the prompt." → enriched_prompt
          ▼
  local model (pipeline._generate): code from enriched_prompt
          ▼
  run_eval(code, row, kind) → pass? ──► return result
          │ fail?
          └─ next hypothesis

  all fail → return first-hypothesis code (best attempt)

Usage:
    from us_recovery_agent import USRecoveryAgent

    agent = USRecoveryAgent(pipeline, model="gpt-4o")
    result = agent.recover(mutated_prompt, row=row, kind="humaneval", entry_point=ep)
    code   = result["code"]
    passed = result["passed"]   # True if any hypothesis passed tests
"""

from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI


# ── Prompts ────────────────────────────────────────────────────────────────────

_HYPO_SYSTEM = """\
You are an expert at diagnosing under-specified programming challenge prompts.
Identify what single constraint was most likely silently removed from the prompt,
making it ambiguous or insufficiently specified for a correct implementation.
"""

_HYPO_USER = """\
The following programming prompt is missing exactly one constraint that was removed.

Typical constraint types that get removed:
  - Numeric bounds         ("1 <= k <= 15", "n <= 10^5", "at least 4 characters")
  - Ordering / tie-breaking ("sorted ascending", "return the first match in case of ties")
  - Output type / format   ("return an integer", "rounded to 2 decimal places", "as a list")
  - Input precondition     ("positive integers only", "non-empty", "starts with zero balance")
  - Edge-case handling     ("return None if empty", "empty sum equals 0", "raise ValueError if negative")
  - Scope limiters         ("all occurrences", "minimum", "first", "distinct", "consecutive", "each")

Prompt:
{prompt}

Return JSON with exactly this shape (3 hypotheses, most likely first):
{{"hypotheses": ["<most likely missing constraint>", "<second>", "<third>"]}}

Each entry is a short phrase naming the missing constraint — do not restate the whole prompt."""

_ENRICH_SYSTEM = """\
You are a technical writer editing a programming challenge prompt.
Your task: reintegrate one missing constraint back into the prompt naturally.
- Keep the existing prompt structure and wording intact.
- Add the constraint where it fits most naturally (usually in the description, not examples).
- Do NOT change the function signature, examples, or any other part.
Return ONLY the enriched prompt text — no JSON, no explanation, no markdown fences."""

_ENRICH_USER = """\
Prompt (missing a constraint):
{prompt}

Constraint to restore: {hypothesis}

Return the complete enriched prompt with the constraint naturally included:"""


# ── Agent ──────────────────────────────────────────────────────────────────────

class USRecoveryAgent:
    """
    Test-guided constraint recovery for US-mutated prompts.

    For each US sample the agent:
      1. Asks GPT-4 to list N likely missing constraints (hypotheses).
      2. For each hypothesis, asks GPT-4 to enrich the prompt.
      3. Generates code from the enriched prompt with the local model.
      4. Runs tests — returns immediately on the first passing attempt.
      5. If all hypotheses fail, returns the first-hypothesis code as best effort.
    """

    def __init__(
        self,
        pipeline,                            # inference_pipeline.FixingPipeline
        openai_client: Optional[OpenAI] = None,
        model:          str              = "gpt-4o",
        max_hypotheses: int              = 3,
    ):
        self.pipeline       = pipeline
        self.client         = openai_client or OpenAI()
        self.model          = model
        self.max_hypotheses = max_hypotheses

    # ── GPT-4 calls ────────────────────────────────────────────────────────────

    def _hypotheses(self, prompt: str) -> list[str]:
        """Ask GPT-4 to list the most likely missing constraints."""
        resp = self.client.chat.completions.create(
            model    = self.model,
            messages = [
                {"role": "system", "content": _HYPO_SYSTEM},
                {"role": "user",   "content": _HYPO_USER.format(prompt=prompt)},
            ],
            response_format = {"type": "json_object"},
            temperature     = 0.2,
        )
        raw = resp.choices[0].message.content
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []

        # model returns {"hypotheses": [...]} or a similar wrapper
        if isinstance(parsed, list):
            return parsed[: self.max_hypotheses]
        for key in ("hypotheses", "constraints", "missing_constraints", "candidates"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key][: self.max_hypotheses]
        # last resort: first list value in the object
        for v in parsed.values():
            if isinstance(v, list):
                return v[: self.max_hypotheses]
        return []

    def _enrich(self, prompt: str, hypothesis: str) -> str:
        """Ask GPT-4 to restore the hypothesised constraint into the prompt."""
        resp = self.client.chat.completions.create(
            model    = self.model,
            messages = [
                {"role": "system", "content": _ENRICH_SYSTEM},
                {"role": "user",   "content": _ENRICH_USER.format(
                    prompt=prompt, hypothesis=hypothesis)},
            ],
            temperature = 0.1,
        )
        return resp.choices[0].message.content.strip()

    # ── Main entry ─────────────────────────────────────────────────────────────

    def recover(
        self,
        prompt:      str,
        row:         dict,            # full dataset row — passed to run_eval
        kind:        str,             # "humaneval" | "mbpp" | "lcb"
        entry_point: Optional[str] = None,
    ) -> dict:
        """
        Returns a dict with:
            code            – generated Python code (best attempt)
            hypothesis      – constraint description used for winning attempt
            enriched_prompt – enriched prompt used for winning attempt
            hypothesis_idx  – index in hypothesis list (−1 if none found)
            passed          – True if any attempt passed the tests
        """
        from eval_fixer import run_eval
        from inference_pipeline import (
            build_lora_instruction,
            extract_code_block,
            extract_func_name,
        )

        hypotheses = self._hypotheses(prompt)

        # No hypotheses → bare pass-through (same as current US behaviour)
        if not hypotheses:
            raw  = self.pipeline._generate(prompt, use_lora=False)
            code = extract_code_block(raw)
            return dict(
                code=code, hypothesis=None, enriched_prompt=prompt,
                hypothesis_idx=-1, passed=run_eval(code, row, kind),
            )

        first: dict = {}

        for idx, hyp in enumerate(hypotheses):
            print(f"    [US agent] hypothesis {idx}: {hyp!r}")

            enriched  = self._enrich(prompt, hyp)
            func_name = entry_point or extract_func_name(enriched)
            instr     = build_lora_instruction(enriched, func_name)
            raw       = self.pipeline._generate(instr, use_lora=False)
            code      = extract_code_block(raw)

            if not first:
                first = dict(
                    code=code, hypothesis=hyp,
                    enriched_prompt=enriched, hypothesis_idx=0,
                )

            if run_eval(code, row, kind):
                print(f"    [US agent] ✓ passed on hypothesis {idx}")
                return dict(
                    code=code, hypothesis=hyp,
                    enriched_prompt=enriched,
                    hypothesis_idx=idx, passed=True,
                )

        print("    [US agent] all hypotheses failed — returning first attempt")
        return {**first, "passed": False}
