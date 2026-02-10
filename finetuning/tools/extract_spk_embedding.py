# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Speaker Embedding Extraction from Training JSONL

This script extracts speaker embeddings directly from training JSONL files.
It groups samples by speaker and uses all audio files for each speaker to compute averaged embeddings.

Usage:
    python extract_speaker_embeddings_from_jsonl.py \
        --model_path pretrained_models/Qwen3-TTS-12Hz-1.7B-Base \
        --train_jsonl train_data.jsonl \
        --output_path speaker_embeddings.pt

Input JSONL format (each line):
{
    "audio": "./data/utt0001.wav",
    "text": "transcript text...",
    "ref_audio": "./data/ref.wav",
    "speaker": "spk_A"
}

The script will:
1. Read all lines from the JSONL file
2. Group by speaker field
3. For each speaker, collect all audio files
4. Extract mel spectrograms from each audio
5. Generate speaker embeddings and average them
6. Save to .pt file
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from tqdm import tqdm


def load_audio_to_np(audio_path: str, target_sr: int = 24000) -> Tuple[np.ndarray, int]:
    """Load and normalize audio file to numpy array.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate (default: 24000)
        
    Returns:
        Tuple of (audio waveform, sampling rate)
    """
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    
    return audio.astype(np.float32), target_sr


@torch.inference_mode()
def extract_mel_spectrogram(audio: np.ndarray, sr: int = 24000) -> torch.Tensor:
    """Extract mel spectrogram from audio waveform.
    
    Args:
        audio: Audio waveform as numpy array
        sr: Sampling rate (default: 24000)
        
    Returns:
        Mel spectrogram tensor of shape [1, time_steps, n_mels]
    """
    assert sr == 24000, "Only support 24kHz audio"
    
    mels = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000
    ).transpose(1, 2)
    
    return mels


@torch.inference_mode()
def extract_speaker_embedding(
    model: Qwen3TTSModel,
    audio_paths: List[str],
    device: str = "cuda"
) -> torch.Tensor:
    """Extract and average speaker embeddings from multiple audio files.
    
    Args:
        model: Qwen3TTSModel instance
        audio_paths: List of paths to audio files
        device: Device to run on (default: "cuda")
        
    Returns:
        Averaged speaker embedding tensor of shape [1, embedding_dim]
    """
    embeddings = []
    
    for audio_path in tqdm(audio_paths, desc="  Processing audios", leave=False):
        try:
            # Load audio
            audio, sr = load_audio_to_np(audio_path)
            
            # Extract mel spectrogram
            mel = extract_mel_spectrogram(audio, sr)
            
            # Extract speaker embedding
            mel = mel.to(device).to(model.model.dtype)
            embedding = model.model.speaker_encoder(mel)
            
            # Store as CPU tensor for averaging
            embeddings.append(embedding.cpu())
        except Exception as e:
            print(f"    Warning: Failed to process {audio_path}: {e}")
            continue
    
    if not embeddings:
        raise ValueError("No valid audio files processed!")
    
    # Stack all embeddings: [num_audios, 1, embedding_dim]
    stacked_embeddings = torch.stack(embeddings)
    
    # Average across all audios: [1, embedding_dim]  
    # This averages embeddings from all audio files for this speaker
    averaged_embedding = stacked_embeddings.mean(dim=0)
    
    print(f"  → Processed {len(embeddings)} audio files, averaged embedding shape: {averaged_embedding.shape}")
    
    return averaged_embedding


def load_and_group_jsonl(jsonl_path: str) -> Dict[str, List[str]]:
    """Load JSONL file and group audio paths by speaker.
    
    Args:
        jsonl_path: Path to training JSONL file
        
    Returns:
        Dictionary mapping speaker ID to list of audio paths
    """
    speaker_to_audios = defaultdict(list)
    
    print(f"Reading JSONL file: {jsonl_path}")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                speaker = data.get('speaker')
                audio = data.get('path')
                
                if not speaker:
                    print(f"  Warning: Line {line_num} missing 'speaker' field, skipping")
                    continue
                if not audio:
                    print(f"  Warning: Line {line_num} missing 'audio' field, skipping")
                    continue
                
                speaker_to_audios[speaker].append(audio)
            except json.JSONDecodeError as e:
                print(f"  Warning: Line {line_num} is not valid JSON: {e}")
                continue
    
    return dict(speaker_to_audios)


def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker embeddings from training JSONL file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/Projects/Qwen3-TTS/pretrained_models/Qwen3-TTS-12Hz-1.7B-Base",
        help="Path to pretrained Qwen3-TTS model"
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default="/data/Projects/Qwen3-TTS/data/finetune/train_2spk.jsonl",
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/Projects/Qwen3-TTS/output",
        help="Output path for speaker embeddings .pt file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Speaker Embedding Extraction from JSONL")
    print("="*60)
    
    # Load and group training data by speaker
    speaker_to_audios = load_and_group_jsonl(args.train_jsonl)
    
    print(f"\nFound {len(speaker_to_audios)} unique speakers:")
    for speaker, audios in speaker_to_audios.items():
        print(f"  - {speaker}: {len(audios)} audio files")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = Qwen3TTSModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.model.to(args.device)
    model.model.eval()
    print("Model loaded successfully!")
    
    # Extract embeddings for each speaker
    print(f"\nExtracting speaker embeddings...")
    speaker_embeddings = {}
    
    for idx, (speaker, audio_paths) in enumerate(speaker_to_audios.items(), 1):
        print(f"\n[{idx}/{len(speaker_to_audios)}] Processing speaker: {speaker}")
        print(f"  Audio files: {len(audio_paths)}")
        
        # Extract and average embeddings from all audio files for this speaker
        embedding = extract_speaker_embedding(
            model=model,
            audio_paths=audio_paths,
            device=args.device
        )
        
        speaker_embeddings[speaker] = embedding
    
    # Save embeddings
    print(f"\nSaving speaker embeddings to {args.output_path}...")
    torch.save(speaker_embeddings, args.output_path)
    
    print("\n" + "="*60)
    print("Speaker Embedding Extraction Complete!")
    print("="*60)
    print(f"Total speakers: {len(speaker_embeddings)}")
    print(f"Output file: {args.output_path}")
    print("\nSpeaker IDs:")
    for speaker in speaker_embeddings.keys():
        print(f"  - {speaker}")
    print("="*60)


if __name__ == "__main__":
    main()