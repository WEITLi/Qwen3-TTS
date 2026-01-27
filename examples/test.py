
import time
import torch
import soundfile as sf
import os 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from qwen_tts import Qwen3TTSModel
def main():
    device = "cuda:0"
    MODEL_PATH = os.path.join(PROJECT_ROOT, "pretrained_models", "Qwen3-TTS-12Hz-1.7B-CustomVoice")

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # 查看支持的语言
    print("Supported languages:", tts.get_supported_languages())
    
    # 查看支持的说话人
    print("Supported speakers:", tts.get_supported_speakers())
if __name__ == "__main__":
    main()
