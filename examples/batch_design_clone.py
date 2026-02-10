import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# create a reference audio in the target style using the VoiceDesign model
design_model = Qwen3TTSModel.from_pretrained(
    "pretrained_models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_text = "您猜怎么着？今儿个天儿不错，咱得去胡同口那家馆子整点儿爆肚，再来二两二锅头，那叫一个地道！"
ref_instruct = "性别: 男性. 音高: 中年低沉音域，由于长期抽烟略带哑嗓. 语速: 语速稍快，吞音明显（连读）. 音量: 音量洪亮，透着股“爷”劲儿. 年龄: 中年. 清晰度: 儿化音极重，个别字词含混但神韵十足. 流畅度: 极其流畅，像是在说单口相声. 口音: 重度北京口音普通话. 音色质感: 粗砺豪放，京味儿十足. 情绪: 惬意满足，侃大山. 语调: 抑扬顿挫，重音突出在“地道”、“爆肚”上. 性格: 局气，爱贫嘴，典型的北京老炮儿."
ref_wavs, sr = design_model.generate_voice_design(
    text=ref_text,
    language="Chinese",
    instruct=ref_instruct
)
sf.write("output/design_clone/voice_design_reference.wav", ref_wavs[0], sr)

# build a reusable clone prompt from the voice design reference
clone_model = Qwen3TTSModel.from_pretrained(
    "pretrained_models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),   # or "voice_design_reference.wav"
    ref_text=ref_text,
)

sentences = [
    "以后机灵点，跟着我好好干，少不了你的好处。",
    "以后这一片我说了算，带你们吃香的喝辣的。",
    "那个废物，他也敢？查清楚了吗？要是让我知道是谁，我非扒了他的皮不可。"
]

# reuse it for multiple single calls

# or batch generate in one call
wavs, sr = clone_model.generate_voice_clone(
    text=sentences,
    language=["Chinese"]*len(sentences),
    voice_clone_prompt=voice_clone_prompt,
)
for i, w in enumerate(wavs):
    sf.write(f"output/design_clone/clone_man_{i}.wav", w, sr)