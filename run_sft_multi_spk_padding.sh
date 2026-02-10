#!/usr/bin/env bash
set -e
source /data/miniconda3/bin/activate qwen3-tts

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="./pretrained_models/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="./pretrained_models/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="./data/finetune/train_2spk.jsonl"
TRAIN_JSONL="./data/finetune/train_with_codes_2spk.jsonl"
OUTPUT_DIR="./output/finetune_2spk"

BATCH_SIZE=4
LR=2e-5
EPOCHS=5


python finetuning/prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python finetuning/sft_multi_spk_padding.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
