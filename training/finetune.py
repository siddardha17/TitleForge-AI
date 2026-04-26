"""
TitleForge AI — LoRA/PEFT Fine-Tuning Script
Base model : TinyLlama/TinyLlama-1.1B-Chat-v1.0
Method     : QLoRA (4-bit NF4 quantization + LoRA adapters)
Trainer    : trl.SFTTrainer
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
ADAPTER_DIR   = os.path.join(OUTPUT_DIR, "lora-adapter")
MERGED_DIR    = os.path.join(OUTPUT_DIR, "merged-model")

# ── Model ──────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LEN   = 512

# ── LoRA Configuration ─────────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r=16,                       # Rank of the LoRA matrices
    lora_alpha=32,              # Scaling factor
    target_modules=[            # Which linear layers to adapt
        "q_proj", "v_proj",
        "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ── Training Hyperparameters ───────────────────────────────────────────────────
TRAINING_ARGS = SFTConfig(
    output_dir=ADAPTER_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    optim="paged_adamw_32bit",
)


def load_jsonl(path: str) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


def setup_model_and_tokenizer():
    """Load quantized base model + tokenizer."""
    print(f"🔄 Loading base model: {BASE_MODEL_ID}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    return model, tokenizer


def train():
    print("=" * 60)
    print("🚀 TitleForge AI — LoRA Fine-Tuning")
    print("=" * 60)

    # 1. Load datasets
    train_dataset = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    eval_dataset  = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    print(f"📦 Train samples: {len(train_dataset)} | Val samples: {len(eval_dataset)}")

    # 2. Load model
    model, tokenizer = setup_model_and_tokenizer()

    # 3. Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TRAINING_ARGS,
    )

    print("\n🏋️  Starting training...")
    trainer.train()

    # 4. Save LoRA adapter
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\n✅ LoRA adapter saved → {ADAPTER_DIR}")

    # 5. Merge adapter into base model
    merge_and_save(model, tokenizer)


def merge_and_save(model, tokenizer):
    """Merge LoRA adapter weights into the full base model and save."""
    print("\n🔀 Merging LoRA adapter into base model...")
    from peft import PeftModel

    # Reload base model in 16-bit for clean merging
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    merged_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged_model = merged_model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"✅ Merged model saved → {MERGED_DIR}")


if __name__ == "__main__":
    train()
