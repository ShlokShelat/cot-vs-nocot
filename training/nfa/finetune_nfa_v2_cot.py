"""
Fine-tuning script — NFA v2 CoT dataset (5-tier)
Verified for: trl==1.3.0, transformers==5.8.0, peft==0.18.1
Model: [BASE_MODEL_NAME]
"""

import os, time, random, csv
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
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

MODEL_PATH = "[BASE_DIR]/[MODEL_SNAPSHOT_PATH]"

TRAIN_FILE = "nfa_v2_cot_train.jsonl"
VAL_FILE   = "nfa_v2_cot_val.jsonl"

OUTPUT_DIR   = "lora_output_nfa_v2_cot_[model_tag]"
DATASET_MODE = "nfa_v2_cot"

MAX_SEQ_LEN      = 8192
EVAL_STEPS       = 200
SAVE_STEPS       = 200
LOG_STEPS        = 10
EVAL_SAMPLES     = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print(f"Mode: NFA-V2-COT | Seed: {SEED} | GPU: {torch.cuda.get_device_name(0)}")
print(f"Model : [BASE_MODEL_NAME]")
print(f"Train : {TRAIN_FILE}")
print(f"Val   : {VAL_FILE}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)


# ===================
# FORMATTER
# ===================
def format_sample(example, tokenizer):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)}

def load_and_format(path, tokenizer, desc):
    print(f"Loading {desc} from {path}...")
    ds = load_dataset("json", data_files=path, split="train")
    print(f"  {len(ds):,} examples. Formatting...")
    return ds.map(
        lambda x: format_sample(x, tokenizer),
        remove_columns=ds.column_names,
        num_proc=4, desc=f"Format {desc}",
    )


# ===================
# CALLBACKS
# ===================
class MetricsLogger(TrainerCallback):
    def __init__(self, output_dir, mode):
        self.path = os.path.join(output_dir, "training_metrics.csv")
        self.mode = mode
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["step","epoch","mode","train_loss",
                                    "eval_loss","learning_rate","grad_norm"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                state.global_step,
                round(state.epoch, 4) if state.epoch else "",
                self.mode,
                logs.get("loss",""), logs.get("eval_loss",""),
                logs.get("learning_rate",""), logs.get("grad_norm",""),
            ])

class TqdmCallback(TrainerCallback):
    def __init__(self): self.pbar = None
    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc="Training", unit="step")
    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar: self.pbar.update(1)
    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar: self.pbar.close()


# ===================
# MAIN
# ===================
def find_last_checkpoint(d):
    if not os.path.isdir(d): return None
    ckpts = sorted([x for x in os.listdir(d) if x.startswith("checkpoint-")],
                   key=lambda x: int(x.split("-")[-1]))
    return os.path.join(d, ckpts[-1]) if ckpts else None


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side     = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN

    train_data = load_and_format(TRAIN_FILE, tokenizer, "train")
    val_data   = load_and_format(VAL_FILE,   tokenizer, "val")

    eval_data = (val_data.shuffle(seed=SEED).select(range(EVAL_SAMPLES))
                 if len(val_data) > EVAL_SAMPLES else val_data)

    train_data = train_data.shuffle(seed=SEED)
    print(f"Train: {len(train_data):,} | Eval: {len(eval_data):,}")

    print("Loading model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, use_cache=False)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("Applying LoRA...")
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                         "gate_proj","up_proj","down_proj"],
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
        bf16=True, tf32=True,
        logging_steps=LOG_STEPS,
        eval_strategy="steps", eval_steps=EVAL_STEPS,
        save_strategy="steps", save_steps=SAVE_STEPS,
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
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=sft_config,
        callbacks=[TqdmCallback(), MetricsLogger(OUTPUT_DIR, DATASET_MODE)],
    )

    last_ckpt = find_last_checkpoint(OUTPUT_DIR)
    if last_ckpt: print(f"Resuming from {last_ckpt}")

    print("\nStarting training...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_ckpt)
    print(f"Training done in {(time.time()-t0)/3600:.2f}h")

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
