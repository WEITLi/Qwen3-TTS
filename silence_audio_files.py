#!/usr/bin/env python3
"""
Process audio files from JSONL:
1. Copy ref_audio files to target directory
2. Add 300ms silence to the end of each wav file
3. Create new JSONL with updated paths
"""

import json
import shutil
import os
from pathlib import Path
from pydub import AudioSegment

# Configuration
SOURCE_JSONL = "/data/Projects/Qwen3-TTS/data/test/spoken.自由聊天风格.prompt_True.jsonl"
TARGET_DIR = "/data/Projects/Qwen3-TTS/data/wavs/4spk"
OUTPUT_JSONL = "/data/Projects/Qwen3-TTS/data/test/spoken.自由聊天风格.prompt_True_with_silence.jsonl"
SILENCE_DURATION_MS = 300

def main():
    # Create target directory if it doesn't exist
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Read the JSONL file
    with open(SOURCE_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    updated_lines = []
    processed_files = set()
    
    for line in lines:
        data = json.loads(line.strip())
        ref_audio_path = data.get('ref_audio', '')
        
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            print(f"Warning: Audio file not found: {ref_audio_path}")
            updated_lines.append(data)
            continue
        
        # Get the filename
        filename = os.path.basename(ref_audio_path)
        target_path = os.path.join(TARGET_DIR, filename)
        
        # Process the audio file only once
        if filename not in processed_files:
            print(f"Processing: {filename}")
            
            try:
                # Load the audio file
                audio = AudioSegment.from_wav(ref_audio_path)
                
                # Create 300ms silence
                silence = AudioSegment.silent(duration=SILENCE_DURATION_MS)
                
                # Append silence to the audio
                audio_with_silence = audio + silence
                
                # Export to target directory
                audio_with_silence.export(target_path, format="wav")
                
                processed_files.add(filename)
                print(f"  ✓ Saved to: {target_path}")
                
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {e}")
                # If processing fails, just copy the original
                shutil.copy2(ref_audio_path, target_path)
        
        # Update the ref_audio path in the data
        data['ref_audio'] = target_path
        updated_lines.append(data)
    
    # Write the updated JSONL
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for data in updated_lines:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Processed {len(processed_files)} unique audio files")
    print(f"✓ Updated JSONL saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
