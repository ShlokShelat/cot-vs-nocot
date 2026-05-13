# Datasets

Datasets are hosted on HuggingFace Hub (anonymised link in root README).
To regenerate from scratch see `../data_generation/`.

## NFA Dataset
- **Task**: Convert a regular expression to an NFA using Thompson's construction
- **Answer**: Markdown transition table with State / Role / symbol / epsilon columns
- **5 tiers** by NFA state count: T1 (4-8), T2 (8-14), T3 (14-22), T4 (22-32), T5 (32-50)
- **Splits**: train / val / test (IID) / ood_test
- **Files**: `nfa_v2_cot_*.jsonl` and `nfa_v2_nocot_*.jsonl`

## DFA Dataset
- **Task**: Convert a regular expression to a minimised DFA
- **Answer**: Markdown transition table with State / Role / symbol columns
  (no epsilon column; `--` for dead/trap state)
- **4 tiers** by minimised DFA state count: T1 (2), T2 (2-4), T3 (3-6), T4 (4-18)
- **Splits**: train / val / test (IID) / ood_test
- **Files**: `dfa_cot_*.jsonl` and `dfa_nocot_*.jsonl`

## Format

Each `.jsonl` line:

```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "Convert regex (a|b)*abb ..."},
    {"role": "assistant", "content": "## Step 1: ..."}
  ],
  "metadata": {
    "regex":    "(a|b)*abb",
    "alphabet": ["a", "b"],
    "tier":     2,
    "hash":     "349dfd84"
  }
}
```

## Loading

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="nfa_v2_cot_train.jsonl", split="train")
```
