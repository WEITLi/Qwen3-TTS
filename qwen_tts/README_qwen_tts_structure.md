# Qwen3-TTS 核心包结构详解

## 概述

`qwen_tts` 是 Qwen3-TTS 项目的核心 Python 包，包含了完整的语音合成系统实现。该包采用模块化设计，分为推理接口、核心模型实现、命令行工具等几个主要部分。

## 目录结构

```
qwen_tts/
├── __init__.py                 # 包初始化文件，导出主要 API
├── __main__.py                 # 包的命令行入口点
├── cli/                        # 命令行界面模块
│   └── demo.py                # Gradio Web UI 演示程序
├── core/                       # 核心模型实现
│   ├── __init__.py            # 核心模块初始化
│   ├── models/                # 主要模型定义
│   │   ├── __init__.py
│   │   ├── configuration_qwen3_tts.py    # 模型配置类
│   │   ├── modeling_qwen3_tts.py         # 模型实现
│   │   └── processing_qwen3_tts.py       # 数据处理器
│   ├── tokenizer_12hz/        # 12Hz 语音分词器
│   │   ├── configuration_qwen3_tts_tokenizer_v2.py
│   │   └── modeling_qwen3_tts_tokenizer_v2.py
│   └── tokenizer_25hz/        # 25Hz 语音分词器
│       ├── configuration_qwen3_tts_tokenizer_v1.py
│       ├── modeling_qwen3_tts_tokenizer_v1.py
│       └── vq/                # 向量量化实现
│           ├── assets/        # 资源文件
│           ├── core_vq.py     # 核心向量量化算法
│           ├── speech_vq.py   # 语音向量量化
│           └── whisper_encoder.py  # Whisper 编码器
└── inference/                 # 推理接口
    ├── qwen3_tts_model.py     # 主要 TTS 模型包装器
    └── qwen3_tts_tokenizer.py # 分词器包装器
```

## 各模块详细说明

### 1. 包入口文件

#### `__init__.py`
- **作用**: 包的主要入口点，定义公共 API
- **功能**: 
  - 导出 `Qwen3TTSModel` 和 `VoiceClonePromptItem` 类
  - 导出 `Qwen3TTSTokenizer` 类
  - 提供统一的包接口

#### `__main__.py`
- **作用**: 包的命令行入口点
- **功能**: 
  - 当使用 `python -m qwen_tts` 时执行
  - 显示可用的 CLI 命令提示
  - 引导用户使用 `qwen-tts-demo` 命令

### 2. 命令行界面 (`cli/`)

#### `demo.py`
- **作用**: Gradio Web UI 演示程序
- **主要功能**:
  - 提供交互式 Web 界面用于语音合成
  - 支持三种模型类型：CustomVoice、VoiceDesign、Base
  - 命令行参数解析和模型加载
  - 支持 HTTPS 部署（Base 模型需要麦克风权限）
- **核心特性**:
  - 自动检测模型类型并提供相应界面
  - 支持批量处理和实时预览
  - 可配置设备、数据类型、注意力机制等参数

### 3. 核心模型实现 (`core/`)

#### `core/__init__.py`
- **作用**: 核心模块的统一导出
- **功能**: 导出两个版本的分词器配置和模型类

#### `core/models/` - 主要模型定义

##### `configuration_qwen3_tts.py`
- **作用**: 定义所有模型的配置类
- **主要配置类**:
  - `Qwen3TTSSpeakerEncoderConfig`: 说话人编码器配置（基于 ECAPA-TDNN）
  - `Qwen3TTSTalkerCodePredictorConfig`: 语音代码预测器配置
  - `Qwen3TTSConfig`: 主模型配置
- **核心参数**:
  - 模型维度、层数、注意力头数
  - RoPE 位置编码参数
  - 滑动窗口注意力配置

##### `modeling_qwen3_tts.py`
- **作用**: 实现完整的 TTS 模型架构
- **主要组件**:
  - `Res2NetBlock`: Res2Net 残差块实现
  - 说话人编码器（ECAPA-TDNN 架构）
  - 语音代码预测器（基于 Transformer）
  - 完整的 TTS 生成模型
- **核心功能**:
  - 多模态输入处理（文本、音频、说话人信息）
  - 流式和非流式生成支持
  - HuggingFace 兼容的模型接口

##### `processing_qwen3_tts.py`
- **作用**: 数据预处理和后处理
- **主要功能**:
  - 文本分词和编码
  - 批量数据处理
  - 输入格式标准化
- **核心类**:
  - `Qwen3TTSProcessor`: 主要处理器类
  - `Qwen3TTSProcessorKwargs`: 处理参数配置

#### `core/tokenizer_12hz/` - 12Hz 语音分词器

##### `configuration_qwen3_tts_tokenizer_v2.py`
- **作用**: 12Hz 分词器的配置定义
- **核心配置**:
  - `Qwen3TTSTokenizerV2DecoderConfig`: 解码器配置
  - `Qwen3TTSTokenizerV2Config`: 主配置类
- **关键参数**:
  - 码本大小：2048
  - 量化器数量：16
  - 采样率：12.5 FPS
  - 上采样率配置

##### `modeling_qwen3_tts_tokenizer_v2.py`
- **作用**: 12Hz 分词器的具体实现
- **主要功能**:
  - 音频编码为离散token
  - token解码为音频波形
  - 多码本向量量化
  - 高质量音频重建

#### `core/tokenizer_25hz/` - 25Hz 语音分词器

##### `configuration_qwen3_tts_tokenizer_v1.py`
- **作用**: 25Hz 分词器的配置定义
- **特点**: 更高的时间分辨率（25 FPS）

##### `modeling_qwen3_tts_tokenizer_v1.py`
- **作用**: 25Hz 分词器的具体实现
- **优势**: 更精细的时间建模能力

#### `core/tokenizer_25hz/vq/` - 向量量化实现

##### `core_vq.py`
- **作用**: 核心向量量化算法实现
- **主要功能**:
  - K-means 聚类初始化
  - 指数移动平均更新
  - 拉普拉斯平滑
  - 向量采样和量化
- **核心算法**:
  - 向量量化（VQ）
  - 残差向量量化（RVQ）
  - 码本学习和更新

##### `speech_vq.py`
- **作用**: 语音专用的向量量化实现
- **功能**: 针对语音信号优化的量化策略

##### `whisper_encoder.py`
- **作用**: Whisper 编码器集成
- **功能**: 利用 Whisper 的语音理解能力进行特征提取

### 4. 推理接口 (`inference/`)

#### `qwen3_tts_model.py` - 主要 TTS 模型包装器

##### 核心类和数据结构

**`Qwen3TTSModel`**: 主要模型包装器类
- 提供 HuggingFace 风格的统一 API
- 支持三种模型类型：CustomVoice、VoiceDesign、Base
- 自动处理模型类型检测和验证

**`VoiceClonePromptItem`**: 语音克隆提示数据结构
```python
@dataclass
class VoiceClonePromptItem:
    ref_code: Optional[torch.Tensor]        # 参考音频代码 (T, Q) 或 (T,)
    ref_spk_embedding: torch.Tensor         # 说话人嵌入 (D,)
    x_vector_only_mode: bool               # 是否仅使用说话人向量
    icl_mode: bool                         # 是否启用上下文学习模式
    ref_text: Optional[str] = None         # 参考文本
```

##### 核心方法详解

**`from_pretrained()`**: 模型加载
- HuggingFace 风格的模型初始化
- 自动注册配置和模型类
- 支持设备映射、数据类型、注意力实现等参数
- 加载生成配置文件 `generate_config.json`

**输入处理和验证**:
- `_normalize_audio_inputs()`: 统一音频输入格式处理
- `_validate_languages()`: 语言支持验证
- `_validate_speakers()`: 说话人支持验证
- `_merge_generate_kwargs()`: 生成参数合并

##### 三种生成模式对比

#### 1. `generate_voice_clone()` - 语音克隆模式

**适用模型**: Base 模型 (0.6B/1.7B-Base)

**核心特性**:
- 基于参考音频进行语音克隆
- 支持两种克隆模式：
  - **ICL 模式** (`x_vector_only_mode=False`): 使用参考音频token + 说话人嵌入 + 参考文本
  - **X-Vector 模式** (`x_vector_only_mode=True`): 仅使用说话人嵌入

**输入参数**:
```python
def generate_voice_clone(
    text: Union[str, List[str]],                    # 目标文本
    language: Union[str, List[str]] = None,         # 语言
    ref_audio: Optional[AudioLike] = None,          # 参考音频
    ref_text: Optional[str] = None,                 # 参考文本（ICL模式必需）
    x_vector_only_mode: Union[bool, List[bool]] = False,  # 克隆模式
    voice_clone_prompt: Optional[...] = None,       # 预构建的克隆提示
    non_streaming_mode: bool = False,               # 非流式模式
    **kwargs                                        # 生成参数
)
```

**处理流程**:
1. 验证模型类型为 "base"
2. 创建或使用预构建的语音克隆提示
3. 构建助手文本：`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
4. 处理参考文本（ICL模式）：`<|im_start|>assistant\n{ref_text}<|im_end|>\n`
5. 生成语音代码并解码为音频
6. 裁剪参考音频部分（ICL模式）

#### 2. `generate_voice_design()` - 语音设计模式

**适用模型**: VoiceDesign 模型 (1.7B-VoiceDesign)

**核心特性**:
- 基于自然语言描述生成语音
- 智能理解语音风格指令
- 支持情感、语调、说话风格控制

**输入参数**:
```python
def generate_voice_design(
    text: Union[str, List[str]],           # 目标文本
    instruct: Union[str, List[str]],       # 语音风格指令
    language: Union[str, List[str]] = None, # 语言
    non_streaming_mode: bool = True,        # 非流式模式
    **kwargs                               # 生成参数
)
```

**处理流程**:
1. 验证模型类型为 "voice_design"
2. 构建助手文本：`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
3. 构建指令文本：`<|im_start|>user\n{instruct}<|im_end|>\n`
4. 生成语音代码并解码为音频

#### 3. `generate_custom_voice()` - 自定义语音模式

**适用模型**: CustomVoice 模型 (0.6B/1.7B-CustomVoice)

**核心特性**:
- 使用预定义的说话人 ID
- 支持 9 种预设音色
- 可选的指令控制（1.7B 模型）

**输入参数**:
```python
def generate_custom_voice(
    text: Union[str, List[str]],              # 目标文本
    speaker: Union[str, List[str]],           # 说话人名称
    language: Union[str, List[str]] = None,   # 语言
    instruct: Optional[Union[str, List[str]]] = None,  # 可选指令（0.6B不支持）
    non_streaming_mode: bool = True,          # 非流式模式
    **kwargs                                  # 生成参数
)
```

**处理流程**:
1. 验证模型类型为 "custom_voice"
2. 验证说话人支持性
3. 构建助手文本：`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
4. 构建指令文本（如果提供）：`<|im_start|>user\n{instruct}<|im_end|>\n`
5. 生成语音代码并解码为音频

##### 三种模式的技术对比

| 特性 | Voice Clone | Voice Design | Custom Voice |
|------|-------------|--------------|--------------|
| **模型要求** | Base 模型 | VoiceDesign 模型 | CustomVoice 模型 |
| **输入复杂度** | 高（需参考音频） | 中（需指令描述） | 低（仅需说话人ID） |
| **灵活性** | 最高（任意音色） | 高（自然语言控制） | 中（预设音色） |
| **计算开销** | 高（特征提取） | 中（指令理解） | 低（直接生成） |
| **质量稳定性** | 依赖参考质量 | 依赖指令质量 | 最稳定 |
| **使用场景** | 个性化克隆 | 创意设计 | 标准化应用 |

##### 辅助方法

**`create_voice_clone_prompt()`**: 创建可重用的克隆提示
- 提取参考音频的语音代码和说话人嵌入
- 支持批量处理
- 返回 `VoiceClonePromptItem` 列表

**`get_supported_speakers()`/`get_supported_languages()`**: 获取支持的说话人和语言列表

#### `qwen3_tts_tokenizer.py` - 语音分词器统一接口

##### 核心类

**`Qwen3TTSTokenizer`**: 语音分词器包装器
- 统一 12Hz 和 25Hz 分词器接口
- 自动检测和适配分词器版本
- 提供 HuggingFace 风格的 API

##### 主要功能

**`from_pretrained()`**: 分词器加载
- 自动注册两个版本的分词器
- 加载特征提取器和模型
- 设备自动检测

**`encode()`**: 音频编码
```python
def encode(
    audios: AudioInput,           # 音频输入
    sr: Optional[int] = None,     # 采样率（numpy输入必需）
    return_dict: bool = True,     # 返回字典格式
)
```

**输出格式**:
- **25Hz 分词器**: `{audio_codes, xvectors, ref_mels}`
- **12Hz 分词器**: `{audio_codes}` (多码本格式)

**`decode()`**: 音频解码
```python
def decode(encoded) -> Tuple[List[np.ndarray], int]
```

**支持的输入格式**:
- ModelOutput（encode 的直接输出）
- 字典格式
- 字典列表格式

##### 音频处理能力

**输入格式支持**:
- 文件路径（本地文件）
- URL（HTTP/HTTPS）
- Base64 编码字符串
- NumPy 数组（需提供采样率）

**自动处理功能**:
- 采样率转换
- 多声道转单声道
- 数据类型标准化
- 批量处理

##### 版本差异

| 特性 | 25Hz 分词器 | 12Hz 分词器 |
|------|-------------|-------------|
| **时间分辨率** | 25 FPS | 12.5 FPS |
| **码本结构** | 单码本 | 16个码本 |
| **输出格式** | (T,) | (T, 16) |
| **额外输出** | xvectors, ref_mels | 无 |
| **适用场景** | 高质量合成 | 实时应用 |

##### 实用方法

**`get_model_type()`**: 获取分词器类型
**`get_input_sample_rate()`**: 获取输入采样率
**`get_output_sample_rate()`**: 获取输出采样率
**`get_encode_downsample_rate()`**: 获取编码下采样率
**`get_decode_upsample_rate()`**: 获取解码上采样率

## 技术架构特点

### 1. 模块化设计
- **分层架构**: 推理接口 → 核心模型 → 底层实现
- **松耦合**: 各模块相对独立，便于维护和扩展
- **标准化**: 遵循 HuggingFace 的设计规范

### 2. 多版本支持
- **12Hz 分词器**: 更高效，适合实时应用
- **25Hz 分词器**: 更精细，适合高质量合成
- **向后兼容**: 自动检测和适配不同版本

### 3. 灵活的输入处理
- **多格式支持**: 文件、URL、base64、numpy 数组
- **批量处理**: 支持单个和批量输入
- **自动转换**: 自动处理采样率转换和格式标准化


## 使用示例

### 基本使用

#### 1. 语音克隆 (Base 模型)
```python
from qwen_tts import Qwen3TTSModel

# 加载 Base 模型
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# 方式1：直接克隆
wavs, sr = model.generate_voice_clone(
    text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a",
    language="English",
    ref_audio="reference.wav",
    ref_text="Hello, this is a reference audio.",
    x_vector_only_mode=False  # 使用 ICL 模式
)

# 方式2：预构建提示（推荐用于批量生成）
prompt_items = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="Hello, this is a reference audio.",
    x_vector_only_mode=False
)

wavs, sr = model.generate_voice_clone(
    text=["Sentence A.", "Sentence B."],
    language=["English", "English"],
    voice_clone_prompt=prompt_items
)
```

#### 2. 语音设计 (VoiceDesign 模型)
```python
# 加载 VoiceDesign 模型
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

# 基于自然语言描述生成语音
wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。"
)

# 英文示例
wavs, sr = model.generate_voice_design(
    text="It's in the top drawer... wait, it's empty? No way, that's impossible!",
    language="English",
    instruct="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
)
```

#### 3. 自定义语音 (CustomVoice 模型)
```python
# 加载 CustomVoice 模型
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

# 使用预设说话人
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese",
    speaker="Vivian",
    instruct="用特别愤怒的语气说"  # 可选的情感控制
)

# 查看支持的说话人
speakers = model.get_supported_speakers()
print(f"Supported speakers: {speakers}")

# 批量生成
wavs, sr = model.generate_custom_voice(
    text=["Hello world!", "你好世界！"],
    language=["English", "Chinese"],
    speaker=["Ryan", "Vivian"],
    instruct=["Very happy.", "温柔地说"]
)
```

#### 4. 语音分词器使用
```python
from qwen_tts import Qwen3TTSTokenizer

# 加载分词器
tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")

# 编码音频
codes = tokenizer.encode("audio.wav")
print(f"Audio codes shape: {codes.audio_codes[0].shape}")

# 解码音频
wavs, sr = tokenizer.decode(codes)

# 支持多种输入格式
codes_url = tokenizer.encode("https://example.com/audio.wav")
codes_base64 = tokenizer.encode("data:audio/wav;base64,...")
codes_numpy = tokenizer.encode(numpy_array, sr=24000)
```

### 命令行使用
```bash
# 启动 CustomVoice Web UI
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8000

# 启动 VoiceDesign Web UI
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --port 8000

# 启动 Base 模型 Web UI（需要 HTTPS）
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 8000 \
  --ssl-certfile cert.pem --ssl-keyfile key.pem

# 使用包模块
python -m qwen_tts
```

### 高级用法

#### 语音设计 + 克隆组合
```python
# 先用 VoiceDesign 创建目标风格的参考音频
design_model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
ref_text = "H-hey! You dropped your... uh... calculus notebook?"
ref_instruct = "Male, 17 years old, tenor range, gaining confidence"

ref_wavs, sr = design_model.generate_voice_design(
    text=ref_text,
    language="English",
    instruct=ref_instruct
)

# 然后用 Base 模型进行克隆
clone_model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),
    ref_text=ref_text,
)

# 使用设计的音色生成新内容
wavs, sr = clone_model.generate_voice_clone(
    text="No problem! I actually... kinda finished those already?",
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
```

## 开发和扩展

### 添加新模型
1. 在 `core/models/` 中定义配置和模型类
2. 在 `inference/` 中创建用户接口
3. 在 `__init__.py` 中导出新的 API

### 添加新功能
1. 核心实现放在 `core/` 中
2. 用户接口放在 `inference/` 中
3. 命令行工具放在 `cli/` 中

### 测试和调试
- 使用 `examples/` 中的示例脚本
- 通过 Web UI 进行交互式测试
- 利用 HuggingFace 的调试工具
