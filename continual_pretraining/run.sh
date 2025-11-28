#!/usr/bin/env bash
## data processsing
# python pretrain_data.py ../ntu_rules_pdfs data/pretrain_corpus.jsonl --pdfplumber --recursive --chunk_words 512 --overlap 50

accelerate launch conti_pretrain.py \
  --data_file data/pretrain_corpus.jsonl \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --output_dir outputs/conti-llama-qlora \
  --use_qlora \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05