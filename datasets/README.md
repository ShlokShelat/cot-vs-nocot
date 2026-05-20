# Datasets

Datasets are included directly in this directory. To regenerate from
scratch see `../data_generation/`.

## NFA Dataset
- **Task**: Convert a regular expression to an NFA using Thompson's construction
- **Answer**: Markdown transition table with State / Role / symbol / epsilon columns
- **5 tiers** by NFA state count: T1 (4-8), T2 (8-14), T3 (14-22), T4 (22-32), T5 (32-50)
- **Size**: 25,000 examples total
- **Splits**: train (~22,500) / val (~1,250) / test IID (1,000) / test OOD (1,000)
- **Files**: `nfa_v2_cot_*.jsonl` and `nfa_v2_nocot_*.jsonl`

## DFA Dataset
- **Task**: Convert a regular expression to a minimised DFA
- **Answer**: Markdown transition table with State / Role / symbol columns
  (no epsilon column; `--` for dead/trap state)
- **4 tiers** by minimised DFA state count: T1 (2), T2 (2-4), T3 (3-6), T4 (4-18)
- **Size**: 25,000 examples total
- **Splits**: train (~22,500) / val (~1,125) / test IID (641) / test OOD (641)
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
ds = load_dataset("json", data_files="datasets/nfa/nfa_v2_cot_train.jsonl", split="train")
```

## IID vs OOD

IID questions use the 5 training templates, all of which name Thompson's
construction explicitly. OOD questions use 8 paraphrased templates that
do not mention Thompson's construction by name. IID[i] and OOD[i] share
the same regex instance and ground-truth answer.
