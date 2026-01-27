#!/usr/bin/env python3
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import time
import torch
import soundfile as sf
import json
import os
import argparse

from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description="Batch inference using Qwen3-TTS VoiceDesign model with JSONL input")
    parser.add_argument("--jsonl_path", type=str, default="/data/Projects/Qwen3-TTS/data/test_custom.jsonl", 
                        help="Path to JSONL input file (default: test_custon.jsonl)")
    parser.add_argument("--output_dir", type=str, default="./output/base-0.6B", 
                        help="Output directory for generated audio files (default: ./output)")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to Qwen3-TTS VoiceDesign model (default: auto-detect)")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to use for inference (default: cuda:0)")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        help="Data type for model (default: bfloat16)")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", 
                        help="Attention implementation (default: eager)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, 
                        help="Maximum number of new tokens to generate (default: 2048)")

    args = parser.parse_args()

    # Set project root and model path
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.model_path is None:
        MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained_models", "Qwen3-TTS-12Hz-0.6B-Base")
    else:
        MODEL_PATH = args.model_path

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=args.device,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    )

    # Load and parse JSONL file
    print(f"Loading data from: {args.jsonl_path}")
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(data)} samples for inference")

    # Prepare batch input
    texts = []
    languages = []
    instructs = []
    keys = []
    speakers = []
    for item in data:
        texts.append(item["text"])
        languages.append(item.get("language", "Auto"))
        speakers.append(item.get("speaker",""))
        instructs.append(item.get("instruct", ""))
        keys.append(item["key"])

    # Perform batch inference
    print("Starting batch inference...")
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_custom_voice(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=instructs,
        max_new_tokens=args.max_new_tokens,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    total_time = t1 - t0
    print(f"Batch inference completed in {total_time:.3f}s")
    print(f"Average time per sample: {total_time / len(data):.3f}s")

    # Save generated audio files
    print(f"Saving audio files to: {os.path.abspath(args.output_dir)}")
    for key, wav in zip(keys, wavs):
        output_path = os.path.join(args.output_dir, f"{key}.wav")
        sf.write(output_path, wav, sr)
        print(f"Saved: {output_path}")

    print(f"All audio files saved successfully!")


if __name__ == "__main__":
    main()
