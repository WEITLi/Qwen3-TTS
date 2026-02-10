"""
混合方案：使用 Base 模型的 speaker_encoder + CustomVoice 模型生成
"""
import torch
import soundfile as sf
import time
import json
from qwen_tts import Qwen3TTSModel
import os
import librosa
import numpy as np
from typing import List, Dict, Any


def load_valid_samples(jsonl_path: str) -> Dict[str, List[Any]]:
    """
    加载 JSONL 文件，只保留有效的样本
    """
    samples = {
        'texts': [],
        'languages': [],
        'speakers': [],
        'instructs': [],
        'keys': [],
        'ref_texts': [],
        'ref_audios': []
    }
    
    skipped_count = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_num} 行 JSON 解析失败: {e}")
                skipped_count += 1
                continue
            
            key = data.get('key', f'line_{line_num}')
            
            # 检查必需字段
            if "ref_audio" not in data or not data["ref_audio"]:
                print(f"⚠️  跳过 {key}: 缺少 ref_audio")
                skipped_count += 1
                continue
            
            if not os.path.exists(data["ref_audio"]):
                print(f"⚠️  跳过 {key}: ref_audio 文件不存在: {data['ref_audio']}")
                skipped_count += 1
                continue
            
            if "ref_text" not in data or not data["ref_text"]:
                print(f"⚠️  跳过 {key}: 缺少 ref_text")
                skipped_count += 1
                continue
            
            if "text" not in data or not data["text"]:
                print(f"⚠️  跳过 {key}: 缺少 text")
                skipped_count += 1
                continue
            
            # 添加样本
            samples['texts'].append(data["text"])
            samples['languages'].append(data.get("language", "Chinese"))
            samples['speakers'].append(data.get("spk", ""))
            samples['instructs'].append(data.get("instruct", ""))
            samples['keys'].append(key)
            samples['ref_texts'].append(data["ref_text"])
            samples['ref_audios'].append(data["ref_audio"])
    
    print(f"\n✅ 从 {jsonl_path} 加载了 {len(samples['texts'])} 个有效样本")
    if skipped_count > 0:
        print(f"⚠️  跳过了 {skipped_count} 个无效样本\n")
    
    return samples


def extract_speaker_embeddings_with_base_model(
    base_model: Qwen3TTSModel,
    ref_audios: List[str],
) -> List[torch.Tensor]:
    """
    使用 Base 模型的 speaker_encoder 提取 x-vectors
    
    Args:
        base_model: Base 模型实例
        ref_audios: 参考音频路径列表
        
    Returns:
        List[torch.Tensor]: x-vector 列表
    """
    print("🎙️  使用 Base 模型提取 speaker embeddings...")
    
    speaker_embeddings = []
    target_sr = base_model.model.speaker_encoder_sample_rate  # 24000
    
    for i, audio_path in enumerate(ref_audios):
        # 加载音频
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        
        # 重采样到 24kHz
        if sr != target_sr:
            wav = librosa.resample(
                y=wav.astype(np.float32),
                orig_sr=int(sr),
                target_sr=target_sr
            )
        
        # 提取 speaker embedding
        spk_emb = base_model.model.extract_speaker_embedding(
            audio=wav,
            sr=target_sr
        )
        speaker_embeddings.append(spk_emb)
        
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(ref_audios)} 个音频")
    
    print(f"✅ 完成 speaker embedding 提取\n")
    return speaker_embeddings


def extract_ref_codes_with_custom_model(
    custom_model: Qwen3TTSModel,
    ref_audios: List[str],
) -> List[torch.Tensor]:
    """
    使用 CustomVoice 模型的 speech_tokenizer 提取 ref_codes
    
    Args:
        custom_model: CustomVoice 模型实例
        ref_audios: 参考音频路径列表
        
    Returns:
        List[torch.Tensor]: ref_code 列表
    """
    print("🎵 使用 CustomVoice 模型提取 ref_codes...")
    
    # 批量编码
    enc = custom_model.model.speech_tokenizer.encode(ref_audios)
    ref_codes = enc.audio_codes
    
    print(f"✅ 完成 ref_code 提取\n")
    return ref_codes


def main():
    device = "cuda:0"
    
    # ========== 步骤 1: 加载 Base 模型（只用于提取 speaker embedding）==========
    print("=" * 80)
    print("步骤 1: 加载 Base 模型（用于提取 speaker embeddings）")
    print("=" * 80)
    
    base_model = Qwen3TTSModel.from_pretrained(
        "/data/Projects/Qwen3-TTS/pretrained_models/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    torch.cuda.synchronize()
    print(f"✅ Base 模型加载完成")
    print(f"   模型类型: {base_model.model.tts_model_type}")
    print(f"   Tokenizer: {base_model.model.tokenizer_type}\n")
    
    # ========== 步骤 2: 加载 CustomVoice 模型（用于生成）==========
    print("=" * 80)
    print("步骤 2: 加载 CustomVoice 模型（用于生成语音）")
    print("=" * 80)
    
    custom_model = Qwen3TTSModel.from_pretrained(
        "/data/Projects/Qwen3-TTS/exp/exp_l50/sft_lr2ef6_8spk_full-1.7B/checkpoint-epoch-3",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    torch.cuda.synchronize()
    print(f"✅ CustomVoice 模型加载完成")
    print(f"   模型类型: {custom_model.model.tts_model_type}")
    print(f"   Tokenizer: {custom_model.model.tokenizer_type}\n")
    
    # ========== 步骤 3: 加载数据 ==========
    print("=" * 80)
    print("步骤 3: 加载数据")
    print("=" * 80)
    
    jsonl_path = "/data/Projects/Qwen3-TTS/data/test/spoken.自由聊天风格.prompt_True.jsonl"
    output_dir = "./output/hybrid-base-speaker-custom-generate"
    os.makedirs(output_dir, exist_ok=True)
    
    samples = load_valid_samples(jsonl_path)
    
    if len(samples['texts']) == 0:
        print("❌ 错误: 没有有效样本可以处理")
        return
    
    # ========== 步骤 4: 提取 speaker embeddings（使用 Base 模型）==========
    print("=" * 80)
    print("步骤 4: 提取 speaker embeddings")
    print("=" * 80)
    
    speaker_embeddings = extract_speaker_embeddings_with_base_model(
        base_model=base_model,
        ref_audios=samples['ref_audios']
    )
    
    # ========== 步骤 5: 提取 ref_codes（使用 CustomVoice 模型）==========
    print("=" * 80)
    print("步骤 5: 提取 ref_codes")
    print("=" * 80)
    
    ref_codes = extract_ref_codes_with_custom_model(
        custom_model=custom_model,
        ref_audios=samples['ref_audios']
    )
    
    # ========== 步骤 6: 构造 voice_clone_prompt ==========
    print("=" * 80)
    print("步骤 6: 构造 voice_clone_prompt")
    print("=" * 80)
    
    voice_clone_prompt = {
        "ref_spk_embedding": speaker_embeddings,
        "ref_code": ref_codes,
        "x_vector_only_mode": [False] * len(samples['texts']),  # ICL 模式
        "icl_mode": [True] * len(samples['texts']),
    }
    print(f"✅ voice_clone_prompt 构造完成\n")
    
    # ========== 步骤 7: 构造 ref_ids ==========
    print("=" * 80)
    print("步骤 7: 构造 ref_ids")
    print("=" * 80)
    
    ref_ids = []
    for ref_text in samples['ref_texts']:
        ref_tok = custom_model._tokenize_texts(
            [custom_model._build_ref_text(ref_text)]
        )[0]
        ref_ids.append(ref_tok)
    print(f"✅ ref_ids 构造完成\n")
    
    # ========== 步骤 8: 生成语音（使用 CustomVoice 模型）==========
    print("=" * 80)
    print("步骤 8: 生成语音")
    print("=" * 80)
    
    # 构造 input_ids
    input_ids = custom_model._tokenize_texts(
        [custom_model._build_assistant_text(t) for t in samples['texts']]
    )
    
    # 构造 instruct_ids
    instruct_ids = []
    for ins in samples['instructs']:
        if ins is None or ins == "":
            instruct_ids.append(None)
        else:
            instruct_ids.append(
                custom_model._tokenize_texts([custom_model._build_instruct_text(ins)])[0]
            )
    
    # 生成参数
    gen_kwargs = custom_model._merge_generate_kwargs(max_new_tokens=128)
    
    print("🎤 开始生成...")
    t0 = time.time()
    
    talker_codes_list, _ = custom_model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt,
        instruct_ids=instruct_ids,
        languages=samples['languages'],
        speakers=samples['speakers'],
        non_streaming_mode=True,
        **gen_kwargs,
    )
    
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"✅ 生成完成，耗时: {t1 - t0:.3f}s\n")
    
    # ========== 步骤 9: 解码并保存 ==========
    print("=" * 80)
    print("步骤 9: 解码并保存音频")
    print("=" * 80)
    
    wavs, sr = custom_model.model.speech_tokenizer.decode(
        [{"audio_codes": c} for c in talker_codes_list]
    )
    
    print("💾 保存音频文件...")
    for i, w in enumerate(wavs):
        key = samples['keys'][i]
        output_path = f"{output_dir}/{key}.wav"
        sf.write(output_path, w, sr)
        print(f"  ✓ {output_path}")
    
    print(f"\n🎉 完成！共生成 {len(wavs)} 个音频文件")
    print(f"📁 输出目录: {output_dir}")
    
    # 清理显存
    del base_model
    torch.cuda.empty_cache()
    print("\n✅ 已释放 Base 模型显存")


if __name__ == "__main__":
    main()
