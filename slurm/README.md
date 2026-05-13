# SLURM Scripts

All scripts target the `long` partition with one H100 NVL GPU (120 GB RAM).
CoT and No-CoT jobs can be submitted in parallel -- they coordinate via
a sentinel file to avoid double dataset generation.

```bash
# NFA
sbatch nfa/submit_nfa_cot_v2.sh      # generates dataset + trains + evaluates
sbatch nfa/submit_nfa_nocot_v2.sh    # waits for dataset + trains + evaluates

# DFA
sbatch dfa/submit_dfa_cot.sh
sbatch dfa/submit_dfa_nocot.sh

# Math
sbatch math/submit_gsm8k.sh
sbatch math/submit_math.sh
sbatch math/run_mathinstruct.sh
```

Edit `BASE`, model path, and venv path variables at the top of each script
to match your cluster layout.
