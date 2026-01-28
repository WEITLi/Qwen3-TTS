#!/usr/bin/env bash
set -e
source /data/miniconda3/bin/activate qwen3-tts

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="./pretrained_models/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="./pretrained_models/Qwen3-TTS-12Hz-1.7B-Base"
<<<<<<< HEAD

RAW_JSONL="./data/finetune_multi_spk/train_raw.jsonl"
TRAIN_JSONL="./data/finetune_multi_spk/train_with_codes.jsonl"
SPEAKER_EMBEDDINGS="./data/finetune_multi_spk/speaker_embeddings.pt"
OUTPUT_DIR="./output/finetune_multi_spk"

BATCH_SIZE=4
LR=2e-5
EPOCHS=3
=======
RAW_JSONL="./data/finetune/train_2spk_full.jsonl"

# exp name
expdir=exp/exp_l50
expname='Qwen3-TTS_sft_2spk_full'
mkdir -p ${expdir}/${expname}

TRAIN_JSONL="./${expdir}/${expname}/train_with_codes.jsonl"
SPEAKER_EMBEDDINGS="./${expdir}/${expname}/speaker_embeddings.pt"
OUTPUT_DIR="./${expdir}/${expname}"

BATCH_SIZE=4
LR=2e-5
EPOCHS=30
>>>>>>> 20309e1 (support multi spk sft)

# Step 1: Prepare data (extract audio codes)
echo "Step 1: Preparing training data..."
python finetuning/prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

# Step 2: Extract speaker embeddings from training data
echo "Step 2: Extracting speaker embeddings..."
<<<<<<< HEAD
python finetuning/extract_speaker_embeddings_from_jsonl.py \
=======
python finetuning/tools/extract_spk_embedding.py \
>>>>>>> 20309e1 (support multi spk sft)
  --model_path ${INIT_MODEL_PATH} \
  --train_jsonl ${TRAIN_JSONL} \
  --output_path ${SPEAKER_EMBEDDINGS} \
  --device ${DEVICE}

# Step 3: Multi-speaker fine-tuning
echo "Step 3: Running multi-speaker fine-tuning..."
<<<<<<< HEAD
python finetuning/sft_12hz_multi_speaker.py \
=======
python finetuning/sft_multi_spk_direct.py \
>>>>>>> 20309e1 (support multi spk sft)
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --speaker_embeddings_path ${SPEAKER_EMBEDDINGS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS}

echo "Multi-speaker fine-tuning completed!"
<<<<<<< HEAD
echo "Output directory: ${OUTPUT_DIR}"
=======
echo "Output directory: ${OUTPUT_DIR}"
>>>>>>> 20309e1 (support multi spk sft)
