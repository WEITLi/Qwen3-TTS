import torch
import soundfile as sf
import time
from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    tts = Qwen3TTSModel.from_pretrained(
        "/data/Projects/Qwen3-TTS/exp/exp_l50/Qwen3-TTS_sft_lr2ef6_2spk_full-1.7B/checkpoint-epoch-3",
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

    texts = ["其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
             "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
             "你是这样的人，我可是不信"
             "你是这样的人，我可是不信"]
    languages = ["Chinese", "Chinese","Chinese","Chinese"]
    speakers = ["青少年女内向_自由聊天","青少年女内向_台词", "青少年男内向_自由聊天","青少年男内向_台词"]
    instructs = ["用伤心的语气说", "用伤心的语气说","用鄙夷的语气说","用鄙夷的语气说"]


    wavs, sr = tts.generate_custom_voice(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=instructs,
        max_new_tokens=128,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        sf.write(f"./output/sft_8spk_full/1.7B_e3-2_{i}.wav", w, sr)


if __name__ == "__main__":
    main()