import torch
import soundfile as sf
import time
import json
from qwen_tts import Qwen3TTSModel
import os
from typing import List, Dict, Any


def load_valid_samples(jsonl_path: str) -> Dict[str, List[Any]]:
    """
    加载 JSONL 文件，只保留有效的样本
    
    Returns:
        包含所有字段列表的字典
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
            
            # 验证必需字段
            key = data.get('key', f'line_{line_num}')
            
            # 检查 ref_audio
            if "ref_audio" not in data or not data["ref_audio"]:
                print(f"⚠️  跳过 {key}: 缺少 ref_audio")
                skipped_count += 1
                continue
            
            if not os.path.exists(data["ref_audio"]):
                print(f"⚠️  跳过 {key}: ref_audio 文件不存在: {data['ref_audio']}")
                skipped_count += 1
                continue
            
            # 检查 ref_text
            if "ref_text" not in data or not data["ref_text"]:
                print(f"⚠️  跳过 {key}: 缺少 ref_text")
                skipped_count += 1
                continue
            
            # 检查 text
            if "text" not in data or not data["text"]:
                print(f"⚠️  跳过 {key}: 缺少 text")
                skipped_count += 1
                continue
            
            # 所有检查通过，添加样本
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


def main():
    device = "cuda:0"
    
    print("🚀 加载模型...")
    tts = Qwen3TTSModel.from_pretrained(
        "/data/Projects/Qwen3-TTS/exp/exp_l50/sft_lr2ef6_8spk_full-1.7B/checkpoint-epoch-3",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    torch.cuda.synchronize()
    print("✅ 模型加载完成\n")

    jsonl_path = "/data/Projects/Qwen3-TTS/data/test/spoken.自由聊天风格.prompt_True.jsonl"
    output_dir = "./output/custom-finetune-1.7B-plain_prompt_true_icl"
    os.makedirs(output_dir, exist_ok=True)

    # 加载有效样本
    print("📂 加载数据...")
    samples = load_valid_samples(jsonl_path)
    
    if len(samples['texts']) == 0:
        print("❌ 错误: 没有有效样本可以处理")
        return

    # 生成语音
    print("🎤 开始生成语音...")
    t0 = time.time()
    wavs, sr = tts.generate_custom_voice_icl(
        text=samples['texts'],
        language=samples['languages'],
        speaker=samples['speakers'],
        instruct=samples['instructs'],
        max_new_tokens=128,
        ref_texts=samples['ref_texts'],    # ← 使用 ref_texts（复数）
        ref_audios=samples['ref_audios']   # ← 使用 ref_audios（复数）
    )
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"✅ 生成完成，耗时: {t1 - t0:.3f}s\n")

    # 保存
    print("💾 保存音频文件...")
    for i, w in enumerate(wavs):
        key = samples['keys'][i]
        output_path = f"{output_dir}/{key}.wav"
        sf.write(output_path, w, sr)
        print(f"  ✓ {output_path}")
    
    print(f"\n🎉 完成！共生成 {len(wavs)} 个音频文件")


if __name__ == "__main__":
    main()
