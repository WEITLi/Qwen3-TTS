import torch
import soundfile as sf
import time
import json
from qwen_tts import Qwen3TTSModel
import os

def main():
    device = "cuda:0"
    tts = Qwen3TTSModel.from_pretrained(
        "/data/Projects/Qwen3-TTS/exp/exp_l50/sft_lr1ef7_8spk_full-0.6B-dropout-cosine-warmup200/checkpoint-epoch-9",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    torch.cuda.synchronize()

    jsonl_path = "/data/Projects/Qwen3-TTS/data/test/sft_dropout.jsonl"
    output_dir = "./output/sft_0.6B_lr1ef7-dropout-consine-e10"

    os.makedirs(output_dir, exist_ok=True)

    texts = []
    languages = []
    speakers = []
    instructs = []
    keys = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                
                if "text" in data:
                    texts.append(data["text"])
                if "language" in data:
                    languages.append(data["language"])
                else:
                    languages.append("Chinese")  

                if "spk" in data:
                    speakers.append(data["spk"])
                if "key" in data:
                    keys.append(data["key"])

                if "instruct" in data:
                    instructs.append(data["instruct"])
                else:
                    instructs.append("")

    print(f"Loaded {len(texts)} samples from {jsonl_path}")


    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=instructs,
        max_new_tokens=128,
        temperature=2.0,
        repetition_penalty=1.1
    )
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")

    # save
    # save 1.{text}.wav; 2.
    # for i, w in enumerate(wavs):
    #     output_path = f"{output_dir}/batch_{i}.wav"
    #     sf.write(output_path, w, sr)
    #     print(f"Saved {output_path}")
    for i, w in enumerate(wavs):
        # # 使用 speaker-text 作为文件名，清理特殊字符
        # speaker = speakers[i] if i < len(speakers) else "unknown"
        # text = texts[i]
        # # 移除或替换文件名中不允许的字符
        # safe_speaker = "".join(str(speaker))
        # safe_text = "".join(text)
        # # 限制文本部分长度
        # safe_text = safe_text[:50] if len(safe_text) > 50 else safe_text
        key = keys[i] if i < len(keys) else "unknown"
        output_path = f"{output_dir}/{key}.wav"
        sf.write(output_path, w, sr)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
