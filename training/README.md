# Training

All scripts use LoRA (r=32, alpha=64) via TRL 1.3.0 SFTConfig.
Edit `MODEL_PATH` at the top of each script before running.

## NFA

```bash
python3 nfa/finetune_nfa_v2_cot.py    # CoT condition
python3 nfa/finetune_nfa_v2_nocot.py  # No-CoT condition
```

## DFA

```bash
python3 dfa/finetune_dfa_cot.py    # CoT condition
python3 dfa/finetune_dfa_nocot.py  # No-CoT condition
```

## Math

```bash
python3 math/finetune_gsm8k_cot.py
python3 math/finetune_gsm8k_nocot.py
python3 math/finetune_math_cot.py
python3 math/finetune_math_nocot.py
python3 math/finetune_mathinstruct_cot.py
python3 math/finetune_mathinstruct_nocot.py
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank r | 32 |
| LoRA alpha | 64 |
| Dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down proj |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup steps | 50 (100 for MathInstruct) |
| Epochs | 2 |
| Batch size | 8 (1 per device × 8 grad accum) |
| Max seq length | 8192 (NFA/DFA), 4096 (math) |
| Precision | bfloat16 |
| Hardware | NVIDIA H100 NVL |
