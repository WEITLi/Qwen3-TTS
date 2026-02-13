#!/usr/bin/env python3
"""
添加特殊 tokens 到 Qwen3-TTS 模型
支持添加如 [laughter], <strong> 等特殊标记
"""

import json
import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import torch
from safetensors.torch import load_file, save_file


def add_special_tokens_to_tokenizer(model_path, new_tokens, output_path=None):
    """
    添加特殊 tokens 到 tokenizer
    
    Args:
        model_path: 模型路径
        new_tokens: 要添加的新 tokens 列表，例如 ['[laughter]', '<strong>', '<weak>']
        output_path: 输出路径，如果为 None 则覆盖原路径
    """
    if output_path is None:
        output_path = model_path
    
    # 1. 加载 tokenizer
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. 添加新的特殊 tokens
    print(f"Adding {len(new_tokens)} new special tokens: {new_tokens}")
    num_added = tokenizer.add_special_tokens({
        'additional_special_tokens': tokenizer.additional_special_tokens + new_tokens
    })
    
    print(f"Successfully added {num_added} tokens")
    print(f"New vocab size: {len(tokenizer)}")
    
    # 3. 保存更新后的 tokenizer
    print(f"Saving tokenizer to {output_path}")
    tokenizer.save_pretrained(output_path)
    
    return tokenizer, num_added


def resize_model_embeddings(model_path, new_vocab_size, output_path=None):
    """
    调整模型的 embedding 层大小以匹配新的 vocab size
    
    Args:
        model_path: 模型路径
        new_vocab_size: 新的词表大小
        output_path: 输出路径
    """
    if output_path is None:
        output_path = model_path
    
    # 如果输出路径不同，先复制所有文件
    if output_path != model_path:
        import shutil
        print(f"\nCopying model files from {model_path} to {output_path}")
        if os.path.exists(output_path):
            print(f"  Output directory already exists, will overwrite files")
        else:
            os.makedirs(output_path, exist_ok=True)
        
        # 复制所有文件（除了我们要修改的）
        for item in os.listdir(model_path):
            src = os.path.join(model_path, item)
            dst = os.path.join(output_path, item)
            
            # 跳过我们要重新生成的文件
            if item in ['model.safetensors', 'config.json']:
                continue
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"  Copied directory: {item}")
            else:
                shutil.copy2(src, dst)
                print(f"  Copied file: {item}")
    
    # 1. 加载模型权重
    model_file = os.path.join(model_path, "model.safetensors")
    print(f"\nLoading model weights from {model_file}")
    state_dict = load_file(model_file)
    
    # 2. 找到需要调整的 embedding 层
    # Qwen3-TTS 的文本 embedding 层
    embedding_key = 'talker.model.text_embedding.weight'
    
    if embedding_key in state_dict:
        old_embeddings = state_dict[embedding_key]
        old_vocab_size, embedding_dim = old_embeddings.shape
        
        print(f"\nResizing {embedding_key}:")
        print(f"  Old shape: {old_embeddings.shape}")
        print(f"  Old vocab size: {old_vocab_size}")
        print(f"  New vocab size: {new_vocab_size}")
        
        if new_vocab_size > old_vocab_size:
            # 需要扩展 embedding
            num_new_tokens = new_vocab_size - old_vocab_size
            
            # 使用正态分布初始化新的 embeddings
            # 使用与原始 embeddings 相同的 mean 和 std
            mean = old_embeddings.mean().item()
            std = old_embeddings.std().item()
            
            new_embeddings = torch.randn(num_new_tokens, embedding_dim, dtype=old_embeddings.dtype)
            new_embeddings = new_embeddings * std + mean
            
            # 拼接旧的和新的 embeddings
            updated_embeddings = torch.cat([old_embeddings, new_embeddings], dim=0)
            state_dict[embedding_key] = updated_embeddings
            
            print(f"  New shape: {updated_embeddings.shape}")
            print(f"  Added {num_new_tokens} new token embeddings")
        elif new_vocab_size == old_vocab_size:
            print(f"  No resize needed (vocab size unchanged)")
        else:
            print(f"  Warning: new_vocab_size < old_vocab_size, skipping resize")
    else:
        print(f"Warning: {embedding_key} not found in model")
    
    # 3. 更新 config.json 中的 text_vocab_size
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 更新 talker_config 中的 text_vocab_size（这才是文本词汇表大小）
    if 'talker_config' in config:
        old_text_vocab_size = config['talker_config'].get('text_vocab_size', 0)
        config['talker_config']['text_vocab_size'] = new_vocab_size
        print(f"\nUpdated talker_config.text_vocab_size: {old_text_vocab_size} -> {new_vocab_size}")
        
        # 注意：vocab_size 是音频 codec 的大小，不应该修改
        codec_vocab_size = config['talker_config'].get('vocab_size', 0)
        print(f"Keeping talker_config.vocab_size (audio codec): {codec_vocab_size}")
    
    # 4. 保存更新后的模型
    os.makedirs(output_path, exist_ok=True)
    
    output_model_file = os.path.join(output_path, "model.safetensors")
    print(f"\nSaving updated model to {output_model_file}")
    save_file(state_dict, output_model_file)
    
    output_config_file = os.path.join(output_path, "config.json")
    print(f"Saving updated config to {output_config_file}")
    with open(output_config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Model embeddings resized successfully!")


def main():
    parser = argparse.ArgumentParser(description="Add special tokens to Qwen3-TTS model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for the updated model (default: overwrite original)"
    )
    parser.add_argument(
        "--tokens",
        type=str,
        nargs='*',
        default=None,
        help='Space-separated list of quoted tokens (e.g., "[laughter]" "<strong>" "</strong>")'
    )
    parser.add_argument(
        "--resize_embeddings",
        action='store_true',
        help="Whether to resize model embeddings (required for training)"
    )
    
    args = parser.parse_args()
    
    # 处理 tokens 参数
    if args.tokens is None or len(args.tokens) == 0:
        # 使用默认值
        new_tokens = ['[laughter]', '<strong>', '</strong>', '<weak>', '</weak>', '[breath]', '[sigh]']
    else:
        # 合并所有参数（可能包含逗号分隔的 tokens）
        import re
        combined = ' '.join(args.tokens)
        
        # 分割（支持逗号、空格、换行符）
        tokens_raw = re.split(r'[,\s]+', combined)
        
        # 清理每个 token：移除引号、逗号、空格和换行符
        new_tokens = []
        for token in tokens_raw:
            cleaned = token.strip().strip('"').strip("'").strip(',').strip()
            if cleaned:
                new_tokens.append(cleaned)
        
        # 去重但保持顺序
        seen = set()
        unique_tokens = []
        for token in new_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        new_tokens = unique_tokens
    
    print("=" * 60)
    print("Adding Special Tokens to Qwen3-TTS Model")
    print("=" * 60)
    print(f"\nTokens to add ({len(new_tokens)}):")
    for i, token in enumerate(new_tokens, 1):
        print(f"  {i}. {token}")
    
    # Step 1: 添加 tokens 到 tokenizer
    tokenizer, num_added = add_special_tokens_to_tokenizer(
        args.model_path,
        new_tokens,
        args.output_path
    )
    
    # Step 2: 如果需要，调整模型 embeddings
    if args.resize_embeddings and num_added > 0:
        print("\n" + "=" * 60)
        print("Resizing Model Embeddings")
        print("=" * 60)
        resize_model_embeddings(
            args.model_path,
            len(tokenizer),
            args.output_path
        )
    elif num_added > 0:
        print("\n⚠️  Warning: Tokens added but embeddings not resized.")
        print("   Use --resize_embeddings flag to resize model embeddings for training.")
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)
    
    # 验证
    print("\nVerification:")
    print(f"  New vocab size: {len(tokenizer)}")
    print(f"  Added tokens: {new_tokens}")
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"    {token} -> ID: {token_id}")


if __name__ == "__main__":
    main()
