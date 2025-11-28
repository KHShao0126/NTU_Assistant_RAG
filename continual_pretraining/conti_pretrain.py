from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
try:
    # optional imports for QLoRA
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    _HAS_PEFT = True
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    BitsAndBytesConfig = None
    _HAS_PEFT = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_dataset(data_file: str, tokenizer, block_size: int = 2048):
    ds = load_dataset("json", data_files=data_file, split="train")

    if "text" not in ds.column_names:
        for col in ds.column_names:
            if col.lower() in ("content", "document", "body"):
                ds = ds.rename_column(col, "text")
                break
        else:
            raise ValueError("Input JSONL must contain a 'text' field per line")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=False,
            return_attention_mask=False,
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    def group_texts(examples):
        concatenated_ids = sum(examples["input_ids"], [])
        total_length = len(concatenated_ids)
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {}
        input_ids = [concatenated_ids[i:i+block_size] for i in range(0, total_length, block_size)]
        return {"input_ids": input_ids, "labels": input_ids.copy()}

    lm_dataset = tokenized.map(group_texts, batched=True, batch_size=2000)
    return lm_dataset


def main(
    data_file: str,
    model_name_or_path: str,
    output_dir: str,
    per_device_train_batch_size: int = 1,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-5,
    block_size: int = 2048,
    use_8bit: bool = False,
    use_qlora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[str] = None,
    gradient_accumulation_steps: int = 1,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    # For CLM, prefer padding with EOS to avoid vocab resize side-effects
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Preparing dataset...")
    train_dataset = prepare_dataset(data_file, tokenizer, block_size=block_size)

    logger.info("Loading model: %s", model_name_or_path)
    model_kwargs = {"trust_remote_code": True}

    # QLoRA path: load in 4-bit and apply LoRA adapters (requires peft + bitsandbytes)
    if use_qlora:
        if not _HAS_PEFT:
            raise ImportError("QLoRA requested but peft/BitsAndBytes are not available. Install 'peft' and 'bitsandbytes'.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # load model in 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        # prepare and wrap with LoRA
        model = prepare_model_for_kbit_training(model)
        target_modules = None
        if lora_target_modules:
            target_modules = [m.strip() for m in lora_target_modules.split(',') if m.strip()]
        else:
            # common target modules for LLaMA-like architectures
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        # Ensure embedding size matches tokenizer (should be unchanged since we didn't add tokens)
        model.resize_token_embeddings(len(tokenizer))
        # Improve memory during training
        try:
            model.config.use_cache = False
        except Exception:
            pass
        # Print trainable params
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    else:
        if use_8bit:
            # lazy import bitsandbytes load flag via transformers
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map="auto", load_in_8bit=True, **model_kwargs
            )
        else:
            # Let HF choose device mapping (works with accelerate)
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16, **model_kwargs)
        # Improve memory during training
        try:
            model.config.use_cache = False
        except Exception:
            pass

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not use_8bit,
        gradient_checkpointing=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        seed=42,
        push_to_hub=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    # If using PEFT/LoRA, save adapter + base model pointers appropriately
    if use_qlora and _HAS_PEFT:
        # Save the peft adapter
        try:
            model.save_pretrained(output_dir)
        except Exception:
            # fallback: use trainer.save_model
            trainer.save_model(output_dir)
        # also save the tokenizer used for training to the output dir so inference can reuse it
        try:
            tokenizer.save_pretrained(output_dir)
        except Exception:
            logger.exception("Failed to save tokenizer to %s", output_dir)
    else:
        trainer.save_model(output_dir)
        # save tokenizer alongside the saved model for consistency
        try:
            tokenizer.save_pretrained(output_dir)
        except Exception:
            logger.exception("Failed to save tokenizer to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="JSONL file with 'text' field per line")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (load model in 4-bit and apply LoRA adapters). Requires 'peft' and 'bitsandbytes'.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated target module names for LoRA (e.g. q_proj,k_proj,v_proj)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(
        data_file=args.data_file,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
        use_8bit=args.use_8bit,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
