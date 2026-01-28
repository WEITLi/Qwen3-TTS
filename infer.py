import torch
import soundfile as sf
import time
from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    tts = Qwen3TTSModel.from_pretrained(
        "/data/Projects/Qwen3-TTS/exp/exp_l50/Qwen3-TTS_sft_2spk/checkpoint-epoch-1",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    torch.cuda.synchronize()
    t0 = time.time()
# wavs, sr = tts.generate_custom_voice(
#     text="这设计师脑洞也太大了，真有人敢这么穿出门啊。",
#     instruct="用愤怒的语气说",
#     speaker="女郎",
# )
    # sf.write("finetuned_test2.wav", wavs[0], sr)

    texts = ["其实我真的有发现，我是一个特别善于观察别人情绪的人。", "你是这样的人，我可是不信"]
    languages = ["Chinese", "Chinese"]
    speakers = ["女郎", "青年男内向"]
    instructs = ["非常开心，刚开始很惊讶", "非常疑惑，用一种嘲讽的语气"]


    wavs, sr = tts.generate_custom_voice(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=instructs,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        sf.write(f"./output/sft_2spk_avg_emb/infer_spk_emo_{i}.wav", w, sr)


if __name__ == "__main__":
    main()