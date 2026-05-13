# Data Generation

## NFA Dataset (5 tiers, 25,000 examples)

```bash
python3 nfa_parallel_dataset_gen_v2.py     --n            25000       --seed         42          --out_prefix   nfa_v2      --test_per_tier 200
```

Output: `nfa_v2_{cot,nocot}_{train,val,test,ood_test}.jsonl`

## DFA Dataset (4 tiers, 25,000 examples)

```bash
python3 dfa_parallel_dataset_gen.py     --n            25000       --seed         42          --out_prefix   dfa         --test_per_tier 200
```

Output: `dfa_{cot,nocot}_{train,val,test,ood_test}.jsonl`

## Verification

Both generators verify L(NFA) = L(DFA) = L(minDFA) on all strings
up to length 7 and silently discard any example that fails.
Examples are deduplicated by MD5 hash of (regex, alphabet).
