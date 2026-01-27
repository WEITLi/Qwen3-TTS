# Qwen3-TTS 仓库架构与结构分析

## 项目概述

Qwen3-TTS 是阿里巴巴 Qwen 团队开发的强大语音生成系列模型，提供全面的语音克隆、语音设计、超高质量人声合成和自然语言语音控制功能。

### 核心特性

- **强大的语音表示能力**: 基于自研的 Qwen3-TTS-Tokenizer-12Hz，实现高效声学压缩和高维语义建模
- **通用端到端架构**: 采用离散多码本 LM 架构，实现全信息端到端语音建模
- **极低延迟流式生成**: 基于创新的双轨混合流式生成架构，端到端合成延迟低至 97ms
- **智能文本理解与语音控制**: 支持自然语言指令驱动的语音生成

## 项目结构

```
Projects/Qwen3-TTS/
├── README.md                    # 项目主文档
├── LICENSE                      # Apache-2.0 许可证
├── pyproject.toml              # Python 项目配置
├── MANIFEST.in                 # 打包配置
├── 
├── qwen_tts/                   # 核心 Python 包
│   ├── __init__.py            # 包初始化，导出主要 API
│   ├── __main__.py            # 命令行入口
│   ├── cli/                   # 命令行界面
│   │   └── demo.py           # Web UI 演示程序
│   ├── core/                  # 核心模型实现
│   │   ├── __init__.py
│   │   ├── models/           # 主要模型定义
│   │   │   ├── configuration_qwen3_tts.py  # 模型配置
│   │   │   ├── modeling_qwen3_tts.py       # 模型实现
│   │   │   └── processing_qwen3_tts.py     # 数据处理
│   │   ├── tokenizer_12hz/   # 12Hz 语音分词器
│   │   │   ├── configuration_qwen3_tts_tokenizer_v2.py
│   │   │   └── modeling_qwen3_tts_tokenizer_v2.py
│   │   └── tokenizer_25hz/   # 25Hz 语音分词器
│   │       ├── configuration_qwen3_tts_tokenizer_v1.py
│   │       ├── modeling_qwen3_tts_tokenizer_v1.py
│   │       └── vq/           # 向量量化相关
│   └── inference/            # 推理接口
│       ├── qwen3_tts_model.py      # 主要 TTS 模型包装器
│       └── qwen3_tts_tokenizer.py  # 分词器包装器
├── 
├── examples/                   # 使用示例
│   ├── test_model_12hz_custom_voice.py    # 自定义语音示例
│   ├── test_model_12hz_voice_design.py    # 语音设计示例
│   ├── test_model_12hz_base.py            # 基础模型示例
│   └── test_tokenizer_12hz.py             # 分词器示例
├── 
├── finetuning/                # 微调相关
│   ├── README.md             # 微调文档
│   ├── dataset.py            # 数据集处理
│   ├── prepare_data.py       # 数据准备脚本
│   └── sft_12hz.py          # 监督微调脚本
├── 
├── assets/                    # 资源文件
│   └── Qwen3_TTS.pdf         # 技术报告
├── 
└── .github/                   # GitHub 配置
    ├── ISSUE_TEMPLATE/       # Issue 模板
    └── workflows/            # CI/CD 工作流
```

## 核心架构组件

### 1. 模型架构层次

#### 1.1 语音分词器 (Speech Tokenizer)
- **Qwen3-TTS-Tokenizer-12Hz**: 12.5 FPS，16 个码本，码本大小 2048
- **Qwen3-TTS-Tokenizer-25Hz**: 25 FPS，多种配置选项
- 支持高质量语音编码和解码，保持语音的语义和声学特征

#### 1.2 主要模型系列

**基础架构组件:**
- `Qwen3TTSSpeakerEncoderConfig`: 说话人编码器配置（基于 ECAPA-TDNN）
- `Qwen3TTSTalkerCodePredictorConfig`: 语音代码预测器配置
- `Qwen3TTSConfig`: 主模型配置

**模型变体:**
1. **CustomVoice 模型** (0.6B/1.7B)
   - 支持 9 种预设音色
   - 多语言支持（中英日韩德法俄葡西意）
   - 指令控制语音风格

2. **VoiceDesign 模型** (1.7B)
   - 基于自然语言描述生成语音
   - 智能语音设计能力
   - 情感和韵律控制

3. **Base 模型** (0.6B/1.7B)
   - 3 秒快速语音克隆
   - 支持微调
   - 跨语言语音生成

### 2. 推理接口设计

#### 2.1 Qwen3TTSModel 类
主要的用户接口，提供统一的 API：

```python
# 主要方法
- from_pretrained()           # HuggingFace 风格模型加载
- generate_custom_voice()     # 自定义语音生成
- generate_voice_design()     # 语音设计生成
- generate_voice_clone()      # 语音克隆生成
- create_voice_clone_prompt() # 创建可重用的克隆提示
```

#### 2.2 Qwen3TTSTokenizer 类
语音分词器接口：

```python
# 主要方法
- from_pretrained()  # 加载分词器
- encode()          # 音频编码为代码
- decode()          # 代码解码为音频
```

### 3. 数据处理流程

#### 3.1 输入格式支持
- **音频输入**: WAV 文件路径、URL、base64 字符串、numpy 数组
- **文本输入**: 纯文本、带语言标记的文本
- **指令输入**: 自然语言描述的语音风格控制

#### 3.2 输出格式
- **音频输出**: numpy 数组列表 + 采样率
- **支持格式**: 24kHz 采样率，float32 格式

## 技术特点

### 1. 模型创新

#### 1.1 双轨混合流式架构
- 支持流式和非流式生成
- 单字符输入即可输出首个音频包
- 端到端延迟低至 97ms

#### 1.2 离散多码本语言模型
- 避免传统 LM+DiT 方案的信息瓶颈
- 消除级联错误
- 提升模型通用性和生成效率

#### 1.3 智能文本理解
- 深度集成文本语义理解
- 自适应调整语调、节奏和情感表达
- 支持多维声学属性控制

### 2. 多语言支持

支持 10 种主要语言：
- 中文（含方言）
- 英语
- 日语
- 韩语
- 德语
- 法语
- 俄语
- 葡萄牙语
- 西班牙语
- 意大利语

### 3. 预设音色

| 音色名称 | 描述 | 母语 |
|---------|------|------|
| Vivian | 明亮、略带锐利的年轻女声 | 中文 |
| Serena | 温暖、温柔的年轻女声 | 中文 |
| Uncle_Fu | 成熟男声，低沉圆润 | 中文 |
| Dylan | 年轻北京男声，清晰自然 | 中文（北京话） |
| Eric | 活泼成都男声，略带沙哑 | 中文（四川话） |
| Ryan | 动感男声，节奏感强 | 英语 |
| Aiden | 阳光美式男声，中音清晰 | 英语 |
| Ono_Anna | 俏皮日式女声，轻盈灵动 | 日语 |
| Sohee | 温暖韩式女声，情感丰富 | 韩语 |

## 使用场景

### 1. 自定义语音生成
```python
wavs, sr = model.generate_custom_voice(
    text="文本内容",
    language="Chinese",
    speaker="Vivian",
    instruct="用愤怒的语气说"
)
```

### 2. 语音设计
```python
wavs, sr = model.generate_voice_design(
    text="文本内容",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声"
)
```

### 3. 语音克隆
```python
wavs, sr = model.generate_voice_clone(
    text="要合成的文本",
    language="English",
    ref_audio="参考音频路径",
    ref_text="参考音频的文本"
)
```

### 4. 语音分词
```python
# 编码
codes = tokenizer.encode("audio.wav")
# 解码
wavs, sr = tokenizer.decode(codes)
```

## 微调支持

### 1. 数据格式
JSONL 格式，每行包含：
```json
{
    "audio": "./data/target.wav",
    "text": "目标音频的文本",
    "ref_audio": "./data/reference.wav"
}
```

### 2. 微调流程
1. 数据准备：提取音频代码
2. 监督微调：使用 SFT 脚本
3. 模型验证：快速推理测试

### 3. 微调脚本
```bash
# 数据准备
python prepare_data.py --input_jsonl train_raw.jsonl --output_jsonl train_with_codes.jsonl

# 开始微调
python sft_12hz.py --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --train_jsonl train_with_codes.jsonl
```


## 部署选项

### 1. 本地部署
- Python 包安装：`pip install qwen-tts`
- 源码安装：支持开发模式
- Web UI：内置 Gradio 界面

### 2. vLLM 支持
- 官方 vLLM-Omni 支持
- 离线推理优化
- 在线服务即将支持

### 3. API 服务
- DashScope API：实时语音合成
- 支持自定义语音、语音克隆、语音设计
- 多区域部署（中国大陆/国际）


### 3. 项目结构说明
- `qwen_tts/core/`: 核心模型实现，包含配置和建模代码
- `qwen_tts/inference/`: 用户友好的推理接口
- `qwen_tts/cli/`: 命令行工具和 Web UI
- `examples/`: 完整的使用示例
- `finetuning/`: 微调工具和文档
