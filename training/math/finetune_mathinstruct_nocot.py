"""
Fine-tuning script — TIGER-Lab/MathInstruct (No-CoT, Answer Only)
=================================================================
Dataset : TIGER-Lab/MathInstruct
Total   : 262,000 examples (single train split — no official test split)
Model   : [BASE_MODEL_NAME]

Split strategy (fixed seed=42, identical in all three scripts):
  shuffle(seed=42)
  [0      :   5000] → held-out test set   (never trained on)
  [5000   : 262000] → training set        (257,000 examples)

No-CoT response format:
  FINAL ANSWER: <extracted answer>
  (no reasoning trace — answer only)

Fields in dataset: instruction (problem), output (solution), source (origin)

Verified for: trl==1.3.0, transformers==5.8.0, peft==0.18.1
"""

import os, time, random, csv, re
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, set_seed
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

# ===================
# CONFIG
# ===================
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"

MODEL_PATH        = "[BASE_DIR]/[MODEL_SNAPSHOT_PATH]"
OUTPUT_DIR        = "lora_output_mathinstruct_nocot_[model_tag]"
DATASET_MODE      = "mathinstruct_nocot_[model_tag]"
MATHINSTRUCT_PATH = "[BASE_DIR]/datasets/MathInstruct/train"

# Held-out test pool carved from the full dataset BEFORE training.
# Must be identical across finetune_mathinstruct_cot_[model_tag].py,
# finetune_mathinstruct_nocot_[model_tag].py, and eval_mathinstruct_[model_tag].py
TEST_HOLD_OUT = 5000

MAX_SEQ_LEN  = 4096
EVAL_STEPS   = 500
SAVE_STEPS   = 500
LOG_STEPS    = 10
# Subsample of hold-out used for eval *during* training (full 5k is slow)
EVAL_SAMPLES_DURING_TRAIN = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
_gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Mode: MathInstruct-No-CoT ([BASE_MODEL_NAME]) | Seed: {SEED} | GPU: {_gpu_name}")
print(f"Dataset: {MATHINSTRUCT_PATH} | Total: 262k")
print(f"Test hold-out: {TEST_HOLD_OUT} | Train: 262k - {TEST_HOLD_OUT} = 257,000")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, then give the final answer clearly at the end. "
    "End your response with: FINAL ANSWER: <your answer>"
)

# ===================
# DATA PREP
# ===================

def extract_answer_from_output(output_str):
    """
    Extract the final answer from a MathInstruct output field.

    MathInstruct answers use several formats:
      1. "The answer is X."          (CoT entries)
      2. "The answer is (A)."        (multiple-choice CoT)
      3. \\boxed{X}                  (MATH-sourced entries)
      4. Last print(...) value       (PoT entries — best effort)
      5. Last number in the string   (final fallback)
    """
    output_str = output_str.strip()

    # 1. "The answer is X"
    m = re.search(
        r"[Tt]he answer is\s*:?\s*\(?([A-Za-z0-9\s\+\-\*/\.\,\/\\\$\%\^\{\}\_\(\)]+?)\)?\.?\s*$",
        output_str,
        re.MULTILINE,
    )
    if m:
        return m.group(1).strip()

    # 2. \boxed{}
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, output_str))
    if matches:
        last_match = matches[-1]
        start = last_match.end()
        depth = 1
        i = start
        while i < len(output_str) and depth > 0:
            if output_str[i] == "{":
                depth += 1
            elif output_str[i] == "}":
                depth -= 1
            i += 1
        return output_str[start:i - 1].strip()

    # 3. PoT: last print() argument
    pot_match = re.findall(r"print\((.+?)\)", output_str)
    if pot_match:
        return pot_match[-1].strip()

    # 4. Fallback: last number
    numbers = re.findall(r"-?\d+\.?\d*", output_str)
    if numbers:
        return numbers[-1]

    # 5. Last resort
    return output_str[-100:].strip()


def make_nocot_response(example):
    """No-CoT: final answer only, no reasoning trace."""
    answer = extract_answer_from_output(example["output"])
    return f"FINAL ANSWER: {answer}"


def format_sample(example, tokenizer):
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": make_nocot_response(example)},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)}


# ===================
# CALLBACKS
# ===================

class MetricsLogger(TrainerCallback):
    def __init__(self, output_dir, mode):
        self.path = os.path.join(output_dir, "training_metrics.csv")
        self.mode = mode
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "mode", "train_loss",
                                    "eval_loss", "learning_rate", "grad_norm"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                state.global_step,
                round(state.epoch, 4) if state.epoch else "",
                self.mode,
                logs.get("loss", ""),
                logs.get("eval_loss", ""),
                logs.get("learning_rate", ""),
                logs.get("grad_norm", ""),
            ])


class TqdmCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc="Training", unit="step")

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()


# ===================
# MAIN
# ===================

def find_last_checkpoint(d):
    if not os.path.isdir(d):
        return None
    ckpts = sorted(
        [x for x in os.listdir(d) if x.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]),
    )
    return os.path.join(d, ckpts[-1]) if ckpts else None


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN

    # ------------------------------------------------------------------
    # Dataset split strategy:
    #   1. Load full 262k dataset from disk
    #   2. shuffle(seed=42)  — same seed in all three scripts
    #   3. [0 : TEST_HOLD_OUT]      → held-out test  (5,000 examples)
    #   4. [TEST_HOLD_OUT : end]    → training set   (257,000 examples)
    # ------------------------------------------------------------------
    print(f"Loading MathInstruct from {MATHINSTRUCT_PATH}...")
    full_ds = load_from_disk(MATHINSTRUCT_PATH)
    print(f"  Full dataset: {len(full_ds)} examples")

    # Shuffle with fixed seed — must match cot + eval scripts
    full_ds = full_ds.shuffle(seed=SEED)

    # Carve held-out test set FIRST (before any training data is touched)
    test_ds_raw  = full_ds.select(range(TEST_HOLD_OUT))
    train_ds_raw = full_ds.select(range(TEST_HOLD_OUT, len(full_ds)))
    print(f"  Train: {len(train_ds_raw)} | Test hold-out: {len(test_ds_raw)}")

    # Subsample the hold-out for eval *during* training (speed)
    eval_during_train_ds_raw = test_ds_raw.select(
        range(min(EVAL_SAMPLES_DURING_TRAIN, len(test_ds_raw))))

    # num_proc=1: tokenizer lambdas are not safely picklable across processes
    print("Formatting training set (this may take a few minutes for 257k examples)...")
    train_ds = train_ds_raw.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=train_ds_raw.column_names,
        num_proc=1,
        desc="Format MathInstruct train (No-CoT)",
    )
    eval_ds = eval_during_train_ds_raw.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=eval_during_train_ds_raw.column_names,
        num_proc=1,
        desc="Format MathInstruct eval (No-CoT)",
    )

    print(f"  Formatted train: {len(train_ds)} | Eval during training: {len(eval_ds)}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    print("Applying LoRA...")
    model = get_peft_model(model, LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        logging_steps=LOG_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        remove_unused_columns=True,
        dataloader_num_workers=4,
        seed=SEED,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
        callbacks=[TqdmCallback(), MetricsLogger(OUTPUT_DIR, DATASET_MODE)],
    )

    last_ckpt = find_last_checkpoint(OUTPUT_DIR)
    if last_ckpt:
        print(f"Resuming from {last_ckpt}")

    print("\nStarting training...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_ckpt)
    print(f"Done in {(time.time() - t0) / 3600:.2f}h")

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
