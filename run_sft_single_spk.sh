#!/usr/bin/env bash
set -e
source /data/miniconda3/bin/activate qwen3-tts

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="./pretrained_models/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="./pretrained_models/Qwen3-TTS-12Hz-1.7B-Base"
# exp name
expdir=exp/exp_l50
expname='Qwen3-TTS_sft_nvlang_single_full-lr2ef6'
mkdir -p ${expdir}/${expname}

RAW_JSONL="./data/finetune/train_nvlang_full.jsonl"
TRAIN_JSONL="./${expdir}/${expname}/train_with_codes.jsonl"
OUTPUT_DIR="./${expdir}/${expname}"



BATCH_SIZE=4
LR=2e-6
EPOCHS=5
SPEAKER_NAME="女郎"

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