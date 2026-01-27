#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import os
import json
import sys
import random

def main():
    # 硬编码默认路径（方便固定使用场景）
    default_excel_file = "/project/tts/tts_expdata/macheng/L50_氛围NPC_AI训练语料/标注完成/文本/纯文本/12.26女郎/女郎-情绪台词.jsonl.xlsx"
    default_audio_folder = "/project/tts/tts_expdata/macheng/L50_氛围NPC_AI训练语料/标注完成/音频/女郎诱惑/情绪台词"
    default_output_jsonl = "/data/Projects/Qwen3-TTS/data/finetune/train_raw.jsonl"
    
    # 支持命令行参数覆盖
    parser = argparse.ArgumentParser(description='Convert Excel file and audio folder to JSONL format for TTS training')
    parser.add_argument('excel_file', nargs='?', default=default_excel_file, help='Path to the Excel file (default: %(default)s)')
    parser.add_argument('audio_folder', nargs='?', default=default_audio_folder, help='Path to the audio folder containing .wav files (default: %(default)s)')
    parser.add_argument('output_jsonl', nargs='?', default=default_output_jsonl, help='Path to the output JSONL file (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.excel_file):
        print(f"Error: Excel file not found: {args.excel_file}")
        sys.exit(1)
    
    if not os.path.exists(args.audio_folder):
        print(f"Error: Audio folder not found: {args.audio_folder}")
        sys.exit(1)
    
    try:
        # Read Excel file
        df = pd.read_excel(args.excel_file)
        
        # Check required columns
        required_columns = ['key', 'text']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Excel file must contain '{col}' column")
                sys.exit(1)
        
        # Get all .wav files in the audio folder
        audio_files = {}
        for filename in os.listdir(args.audio_folder):
            if filename.endswith('.wav'):
                # Extract numeric part from filename (e.g., 01.wav -> 1)
                try:
                    # Remove .wav extension and leading zeros
                    file_num = int(os.path.splitext(filename)[0].lstrip('0'))
                    audio_files[file_num] = os.path.abspath(os.path.join(args.audio_folder, filename))
                except ValueError:
                    print(f"Warning: Skipping file with invalid name: {filename}")
                    continue
        
        # Select a random reference audio from available audio files
        if audio_files:
            random_key = random.choice(list(audio_files.keys()))
            ref_audio_path = audio_files[random_key]
            print(f"Selected reference audio: {ref_audio_path}")
        else:
            print("Error: No audio files found in the specified folder")
            sys.exit(1)
        
        # Prepare output data
        output_data = []
        for _, row in df.iterrows():
            key = row['key']
            text = row['text']
            
            # Check if key is numeric
            if not isinstance(key, (int, float)):
                print(f"Warning: Skipping row with non-numeric key: {key}")
                continue
            
            key_int = int(key)
            
            # Check if corresponding audio file exists
            if key_int in audio_files:
                output_data.append({
                    "audio": audio_files[key_int],
                    "text": str(text).strip(),
                    "ref_audio": ref_audio_path
                })
            else:
                print(f"Warning: No corresponding audio file for key {key_int}")
        
        # Write to JSONL file
        with open(args.output_jsonl, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Successfully generated {args.output_jsonl} with {len(output_data)} entries")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()