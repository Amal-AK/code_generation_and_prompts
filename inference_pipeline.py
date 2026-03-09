"""
inference_pipeline.py
─────────────────────
Fixing pipeline:

  prompt
    ├─ LoRA classifier ──→ CLEAN / US / low-conf ──→ base model (fixer LoRA OFF) ──→ code
    └─ LV / SF (conf ≥ threshold)                ──→ fixer LoRA ON              ──→ code

US is handled externally; pipeline treats it as pass-through (no fixer LoRA).

Usage:
    pipeline = FixingPipeline.from_checkpoints(
        classifier_ckpt  = "lora_1b_output/best_lora_classifier.pt",
        lora_adapter_dir = "finetune_lv_sf_output/best_lora_sft",
    )
    result = pipeline(prompt)
    print(result["code"])
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_lora_classifier import LoRAPromptClassifier

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_MAP             = {0: "LV", 1: "SF", 2: "US", 3: "CLEAN"}
CLF_MODEL_NAME        = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CLF_HIDDEN_DIM        = 1536
DEFAULT_GEN_MODEL     = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_THRESHOLD     = 0.75
DEFAULT_MAX_NEW       = 512


# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_code_block(text: str) -> str:
    """Return first ```python ... ``` block, or raw text as fallback."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def extract_func_name(prompt: str) -> Optional[str]:
    m = re.search(r"\bdef\s+(\w+)\s*\(", prompt)
    return m.group(1) if m else None


def build_lora_instruction(mutated_prompt: str, func_name: Optional[str]) -> str:
    func_line = (
        f"Write **one** function named `{func_name}` that solves the task."
        if func_name else
        "Write **one** Python function that solves the task."
    )
    return textwrap.dedent(f"""
        You are a senior Python developer.

        Task:
        {mutated_prompt}

        {func_line}
        If helpers are needed, define them above the main function.

        **Use only the Python standard library and place every required `import` at the very top.**

        Return *only* valid Python code in a single code block:
        ```python
        <your code here>
        ```
    """).strip()


# ── Pipeline ───────────────────────────────────────────────────────────────────
class FixingPipeline:
    def __init__(
        self,
        classifier:    LoRAPromptClassifier,
        clf_tokenizer,
        gen_model,
        gen_tokenizer,
        clf_device:          str   = "cuda:0",
        confidence_threshold: float = DEFAULT_THRESHOLD,
        max_new_tokens:       int   = DEFAULT_MAX_NEW,
    ):
        self.classifier           = classifier
        self.clf_tokenizer        = clf_tokenizer
        self.gen_model            = gen_model
        self.gen_tokenizer        = gen_tokenizer
        self.clf_device           = clf_device
        self.confidence_threshold = confidence_threshold
        self.max_new_tokens       = max_new_tokens

    # ── Factory ────────────────────────────────────────────────────────────────
    @classmethod
    def from_checkpoints(
        cls,
        classifier_ckpt:      str,
        lora_adapter_dir:     str,
        gen_model_name:       str   = DEFAULT_GEN_MODEL,
        clf_model_name:       str   = CLF_MODEL_NAME,
        clf_device:           str   = "cuda:0",
        confidence_threshold: float = DEFAULT_THRESHOLD,
        max_new_tokens:       int   = DEFAULT_MAX_NEW,
    ) -> "FixingPipeline":

        # ── LoRA classifier ────────────────────────────────────────────────────
        print(f"Loading classifier: {clf_model_name}")
        ckpt = torch.load(classifier_ckpt, map_location=clf_device, weights_only=False)

        clf_tok     = AutoTokenizer.from_pretrained(clf_model_name, trust_remote_code=True)
        base_encoder = AutoModel.from_pretrained(
            clf_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(clf_device)

        # Re-apply the same LoRA config that was used during training
        lora_cfg_dict = ckpt.get("lora_config", {}) or {}
        lora_config   = LoraConfig(
            r               = lora_cfg_dict.get("r", 16),
            lora_alpha      = lora_cfg_dict.get("lora_alpha", 32),
            lora_dropout    = lora_cfg_dict.get("lora_dropout", 0.05),
            target_modules  = lora_cfg_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias            = "none",
        )
        encoder    = get_peft_model(base_encoder, lora_config)
        classifier = LoRAPromptClassifier(encoder, hidden_dim=CLF_HIDDEN_DIM, num_classes=4).to(clf_device)
        classifier.load_state_dict(ckpt["model_state_dict"], strict=False)
        classifier.eval()
        print(f"  Classifier loaded  →  val_acc={ckpt.get('val_acc', '?'):.4f}")

        # ── Generation model + fixer LoRA ──────────────────────────────────────
        print(f"Loading generation model: {gen_model_name}")
        gen_tok = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=True)
        if gen_tok.pad_token_id is None:
            gen_tok.pad_token_id = gen_tok.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            torch_dtype  = torch.float16,
            device_map   = "auto",
            trust_remote_code = True,
        )
        gen_model = PeftModel.from_pretrained(base, lora_adapter_dir)
        gen_model.eval()
        print(f"  Fixer LoRA loaded  ←  {lora_adapter_dir}")

        return cls(
            classifier           = classifier,
            clf_tokenizer        = clf_tok,
            gen_model            = gen_model,
            gen_tokenizer        = gen_tok,
            clf_device           = clf_device,
            confidence_threshold = confidence_threshold,
            max_new_tokens       = max_new_tokens,
        )

    # ── Classify ───────────────────────────────────────────────────────────────
    def classify(self, prompt: str) -> tuple[str, float]:
        enc = self.clf_tokenizer(
            [prompt], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self.clf_device)
        with torch.no_grad():
            logits = self.classifier(enc["input_ids"], enc["attention_mask"])
        probs    = torch.softmax(logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        return LABEL_MAP[pred_idx], probs[pred_idx].item()

    # ── Generate ───────────────────────────────────────────────────────────────
    def _generate(self, instruction: str, use_lora: bool) -> str:
        if use_lora:
            self.gen_model.enable_adapter_layers()
        else:
            self.gen_model.disable_adapter_layers()

        text   = self.gen_tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.gen_tokenizer(text, return_tensors="pt").to(
            next(self.gen_model.parameters()).device
        )
        with torch.no_grad():
            output = self.gen_model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                do_sample      = False,
                pad_token_id   = self.gen_tokenizer.eos_token_id,
            )
        return self.gen_tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    # ── Main entry ─────────────────────────────────────────────────────────────
    def __call__(self, prompt: str, entry_point: Optional[str] = None) -> dict:
        """
        Args:
            prompt       : the (possibly mutated) problem statement
            entry_point  : optional function name hint

        Returns:
            code          : generated Python code
            mutation_type : classifier prediction (LV / SF / US / CLEAN)
            confidence    : classifier confidence
            lora_used     : whether the fixer LoRA was active
        """
        mut_type, confidence = self.classify(prompt)
        use_lora = mut_type == "LV" and confidence >= self.confidence_threshold

        if use_lora:
            func_name   = entry_point or extract_func_name(prompt)
            instruction = build_lora_instruction(prompt, func_name)
        else:
            instruction = prompt

        raw  = self._generate(instruction, use_lora=use_lora)
        code = extract_code_block(raw)

        return {
            "code":          code,
            "mutation_type": mut_type,
            "confidence":    confidence,
            "lora_used":     use_lora,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierCkpt",  required=True)
    parser.add_argument("--loraAdapterDir",  required=True)
    parser.add_argument("--genModel",        default=DEFAULT_GEN_MODEL)
    parser.add_argument("--threshold",       type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--inputFile",       required=True,
                        help="JSONL with fields: prompt, [entry_point]")
    parser.add_argument("--outputFile",      required=True)
    args = parser.parse_args()

    pipeline = FixingPipeline.from_checkpoints(
        classifier_ckpt      = args.classifierCkpt,
        lora_adapter_dir     = args.loraAdapterDir,
        gen_model_name       = args.genModel,
        confidence_threshold = args.threshold,
    )

    with open(args.inputFile) as fin, open(args.outputFile, "w") as fout:
        for line in fin:
            rec    = json.loads(line)
            result = pipeline(
                prompt      = rec["prompt"],
                entry_point = rec.get("entry_point"),
            )
            fout.write(json.dumps({**rec, **result}) + "\n")
            print(
                f"[{result['mutation_type']} conf={result['confidence']:.2f} "
                f"lora={result['lora_used']}] {rec.get('entry_point', '?')}"
            )
