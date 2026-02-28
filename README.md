# Code Generation & Prompts

A framework for studying the robustness of code generation models through prompt mutation.

## Overview

The pipeline has three main stages:

1. **Mutation generation** — perturb coding prompts from HumanEval, MBPP, and LiveCodeBench using three mutation strategies:
   - **LV** (Lexical Variation): surface-level rephrasing
   - **SF** (Semantic Flip): meaning-altering changes
   - **US** (Under-Specification): removing key details

2. **Inference** — run code generation models (Qwen2.5-Coder, DeepSeek-Coder, CodeLlama, Devstral) on original and mutated prompts.

3. **Judging** — score each mutation on recoverability, naturalness, lexical compliance, and semantic preservation using `Qwen2.5-Coder-32B-Instruct` as a local judge.

## Key Scripts

| Script | Description |
|---|---|
| `generate_mutants.py` | Generate mutated prompts |
| `main_inference.py` | Run models on original/mutated prompts |
| `mutation_judge.py` | Score mutations with a local LLM judge |
| `train_classifier.py` | Train a classifier on judge scores |
| `train_lora_classifier.py` | LoRA-based classifier variant |
| `train_full_classifier.py` | Full fine-tuned classifier variant |

## Datasets

- [HumanEval](https://github.com/openai/human-eval)
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- [LiveCodeBench](https://livecodebench.github.io/)

## Usage

```bash
# 1. Generate mutations
python generate_mutants.py

# 2. Run inference
bash run_inference.sh

# 3. Judge mutations
bash run_judge.sh

# 4. Train classifier
bash run_full_classifier.sh
```
