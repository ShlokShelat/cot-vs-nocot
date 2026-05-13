# Data Generation

## NFA Dataset (5 tiers, 25,000 examples)

```bash
python3 nfa_parallel_dataset_gen_v2.py \
    --n 25000 --seed 42 --out_prefix nfa_v2 --test_per_tier 200
```

Output: `nfa_v2_{cot,nocot}_{train,val,test,ood_test}.jsonl`

## DFA Dataset (4 tiers, 25,000 examples)

```bash
python3 dfa_parallel_dataset_gen.py \
    --n 25000 --seed 42 --out_prefix dfa --test_per_tier 200
```

Output: `dfa_{cot,nocot}_{train,val,test,ood_test}.jsonl`

## Verification

Both generators verify L(NFA) = L(DFA) = L(minDFA) on all strings up to
length 7 and silently discard any example that fails.
Examples are deduplicated by MD5 hash of (regex, alphabet).

## OOD Question Templates

Both datasets include OOD test splits using 8 paraphrased question templates
that differ structurally from the 5 IID training templates:
- No mention of Thompson's construction by name
- No backtick-formatted regex
- Varied framing (informal, homework-problem, formal mathematical)
- Some templates focus on specific components rather than the full construction

The IID and OOD test sets are perfectly aligned: OOD[i] and IID[i]
use the same regex instance and ground-truth answer.
