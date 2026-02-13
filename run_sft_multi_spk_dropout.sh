#!/bin/bash

# 配置参数
DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="./pretrained_models/Qwen3-TTS-Tokenizer-12Hz"
BASE_MODEL_PATH="pretrained_models/Qwen3-TTS-12Hz-0.6B-Base"  
INIT_MODEL_PATH="pretrained_models/Qwen3-TTS-12Hz-0.6B-Base-Extended"  
RAW_JSONL="./data/finetune/train_8spk_rich.jsonl"

# 输出路径
expdir=exp/exp_l50
expname='sft_lr1ef7_8spk_full-0.6B-dropout-cosine-warmup200'
mkdir -p ${expdir}/${expname}

TRAIN_JSONL="./${expdir}/${expname}/train_with_codes.jsonl"
SPEAKER_EMBEDDINGS="./${expdir}/${expname}/speaker_embeddings.pt"
OUTPUT_DIR="./${expdir}/${expname}"

# 训练参数
BATCH_SIZE=4
LEARNING_RATE=1e-7
NUM_EPOCHS=10
TOKEN_DROPOUT=0.15
LR_SCHEDULER="cosine_with_warmup"
COSINE_MIN_LR=0.1

echo "========================================"
echo "训练配置:"
echo "========================================"
echo "  设备: $DEVICE"
echo "  Tokenizer模型: $TOKENIZER_MODEL_PATH"
echo "  基础模型（提取embedding）: $BASE_MODEL_PATH"
echo "  训练模型（扩展后）: $INIT_MODEL_PATH"
echo "  原始数据: $RAW_JSONL"
echo "  输出目录: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Token dropout: $TOKEN_DROPOUT"
echo "========================================"
echo ""

# Step 1: Prepare data (extract audio codes)
echo "Step 1: Preparing training data..."
python finetuning/prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

echo "✓ Step 1 完成"
echo ""

# Step 2: Extract speaker embeddings from training data
# 注意：使用基础模型（未扩展）来提取 speaker embeddings
echo "Step 2: Extracting speaker embeddings..."
echo "  使用基础模型: $BASE_MODEL_PATH"
python finetuning/tools/extract_spk_embedding.py \
  --model_path ${BASE_MODEL_PATH} \
  --train_jsonl ${TRAIN_JSONL} \
  --output_path ${SPEAKER_EMBEDDINGS} \
  --device ${DEVICE}

echo "✓ Step 2 完成"
echo ""

# Step 3: 开始训练
# 注意：使用扩展后的模型进行训练
echo "Step 3: Starting training..."
echo "  使用扩展模型: $INIT_MODEL_PATH"
python finetuning/sft_with_token_dropout_0.6B.py \
    --init_model_path "${INIT_MODEL_PATH}" \
    --output_model_path "${OUTPUT_DIR}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --speaker_embeddings_path "${SPEAKER_EMBEDDINGS}" \
    --batch_size ${BATCH_SIZE} \
    --lr_scheduler ${LR_SCHEDULER} \
    --cosine_min_lr ${COSINE_MIN_LR} \
    --lr ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --token_dropout ${TOKEN_DROPOUT} \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --save_steps 500 \
    --logging_steps 10

echo ""
echo "========================================"
echo "✓ 训练完成!"
echo "========================================"
