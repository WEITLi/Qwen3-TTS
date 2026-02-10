from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "/data/Projects/Qwen3-TTS/exp/exp_l50/sft_lr1ef6_8spk_full-1.7B/checkpoint-epoch-9",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# 检查 speaker_encoder 是否存在
print(f"speaker_encoder: {model.model.speaker_encoder}")
print(f"是否为 None: {model.model.speaker_encoder is None}")
