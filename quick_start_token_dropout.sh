#!/bin/bash

set -e  

# 模型路径
BASE_MODEL="pretrained_models/Qwen3-TTS-12Hz-0.6B-Base"
EXTENDED_MODEL="pretrained_models/Qwen3-TTS-12Hz-0.6B-Base-Extended"

# 特殊 tokens（使用列表格式，支持逗号分隔，每行可以多个 token）
SPECIAL_TOKENS=(
    "<strong>", "</strong>", "[noise]",
    "[laughter]", "<laughter>", "</laughter>",
    "[cough]", "[clucking]", "[accent]",
    "[quick_breath]", "[breath]", "[hissing]",
    "[sigh]", "[vocalized-noise]", "[lipsmack]",
    "[mn]", "[throat]", "[pause]",
    "<sustain>", "</sustain>",
    "<speak_fast>", "</speak_fast>",
    "<speak_slowly>", "</speak_slowly>",
    "<tone_sandhi>", "</tone_sandhi>",
) 
# ============  添加特殊 Tokens ============
echo ""
echo " 添加特殊 tokens 到模型..."
echo "----------------------------------------"

if [ -d "$EXTENDED_MODEL" ]; then
    echo "⚠️  扩展模型已存在: $EXTENDED_MODEL"
    read -p "是否覆盖? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过添加 tokens 步骤"
    else
        python add_special_tokens.py \
            --model_path "$BASE_MODEL" \
            --output_path "$EXTENDED_MODEL" \
            --tokens "${SPECIAL_TOKENS[@]}" \
            --resize_embeddings
    fi
else
    python finetuning/tools/add_special_tokens.py \
        --model_path "$BASE_MODEL" \
        --output_path "$EXTENDED_MODEL" \
        --tokens "${SPECIAL_TOKENS[@]}" \
        --resize_embeddings
fi

echo "✓ 特殊 tokens 添加完成"
