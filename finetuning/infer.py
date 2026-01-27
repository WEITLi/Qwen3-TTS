import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "/data/Projects/Qwen3-TTS/output/finetune/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="这设计师脑洞也太大了，真有人敢这么穿出门啊。",
    instruct="用愤怒的语气说",
    speaker="speaker_1",
)
sf.write("finetuned_test2.wav", wavs[0], sr)