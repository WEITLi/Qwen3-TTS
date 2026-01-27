#!/usr/bin/env bash
set -e
source /data/miniconda3/bin/activate qwen3-tts

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="./pretrained_models/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="./pretrained_models/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="./data/finetune/train_raw.jsonl"
TRAIN_JSONL="./data/finetune/train_with_codes.jsonl"
OUTPUT_DIR="./output/finetune"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="speaker_1"

python finetuning/prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python finetuning/sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}