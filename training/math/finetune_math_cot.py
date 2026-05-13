"""
Fine-tuning script — Hendrycks MATH CoT
=========================================
Train: 7500 examples (lighteval/MATH train split)
Test:  5000 examples (lighteval/MATH test split)

CoT response format:
  Full solution field (reasoning steps + \\boxed{answer})
  FINAL ANSWER: <extracted boxed answer>

Fields in dataset: problem, solution, level (Level 1-5), type (subject)

Dataset: DigitalLearningGmbH/MATH-lighteval (pre-downloaded to disk)

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

MODEL_PATH   = "[BASE_DIR]/[MODEL_SNAPSHOT_PATH]"
OUTPUT_DIR   = "lora_output_math_cot_[model_tag]"
DATASET_MODE = "math_cot"
MATH_PATH    = "[BASE_DIR]/datasets/lighteval_MATH"

MAX_SEQ_LEN  = 4096
EVAL_STEPS   = 200
SAVE_STEPS   = 200
LOG_STEPS    = 10
EVAL_SAMPLES = 200   # subsample eval for speed during training

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
_gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Mode: MATH-CoT | Seed: {SEED} | GPU: {_gpu_name}")
print(f"Dataset: {MATH_PATH} | Train: 7500 | Test: 5000")
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

def extract_boxed_answer(solution_str):
    """
    Extract the content inside the last \\boxed{...} in the solution.
    Handles nested braces, e.g. \\boxed{\\frac{1}{2}}.
    Falls back to the last number found if no \\boxed is present.
    """
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, solution_str))
    if not matches:
        numbers = re.findall(r"-?\d+\.?\d*", solution_str)
        return numbers[-1] if numbers else solution_str.strip()

    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(solution_str) and depth > 0:
        if solution_str[i] == "{":
            depth += 1
        elif solution_str[i] == "}":
            depth -= 1
        i += 1
    return solution_str[start:i - 1].strip()


def make_cot_response(example):
    """
    CoT: full solution (reasoning trace + boxed answer) followed by FINAL ANSWER marker.
    The solution field already contains the complete step-by-step reasoning.
    """
    solution = example["solution"].strip()
    answer   = extract_boxed_answer(solution)
    return f"{solution}\nFINAL ANSWER: {answer}"


def format_sample(example, tokenizer):
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["problem"]},
        {"role": "assistant", "content": make_cot_response(example)},
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

    print(f"Loading dataset from {MATH_PATH}...")
    train_ds = load_from_disk(f"{MATH_PATH}/train")
    test_ds  = load_from_disk(f"{MATH_PATH}/test")
    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")

    # Shuffle before formatting for proper reproducibility
    train_ds = train_ds.shuffle(seed=SEED)
    test_ds  = test_ds.shuffle(seed=SEED)

    # num_proc=1: tokenizer lambdas are not safely picklable across processes
    train_ds = train_ds.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=train_ds.column_names,
        num_proc=1,
        desc="Format MATH train (CoT)",
    )
    test_ds = test_ds.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=test_ds.column_names,
        num_proc=1,
        desc="Format MATH eval (CoT)",
    )

    # Subsample eval for speed during training
    eval_ds = test_ds.select(range(min(EVAL_SAMPLES, len(test_ds))))
    print(f"  Eval (subsampled): {len(eval_ds)}")

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
        warmup_steps=50,
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
