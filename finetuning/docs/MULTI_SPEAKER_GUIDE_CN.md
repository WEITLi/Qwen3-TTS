# 多说话人微调使用指南

## 概述

此实现支持从训练数据 JSONL 文件中提取多说话人的 speaker embeddings，并用于微调。

**关键特性**：
- ✅ 原始文件 `dataset.py` 和 `sft_12hz.py` 保持不变
- ✅ 新文件 `dataset_multi_speaker.py` 和 `sft_12hz_multi_speaker.py` 用于多说话人训练
- ✅ 使用训练数据中每个说话人的**所有 audio 文件**提取 embedding
- ✅ 支持同一批次中包含不同说话人的样本

## 使用流程

### 步骤 1: 准备训练数据

训练数据格式（JSONL，每行一个样本）：

```json
{"audio": "./data/spk_A/utt0001.wav", "text": "其实我真的有发现，我是一个特别善于观察别人情绪的人。", "ref_audio": "./data/spk_A/ref.wav", "speaker": "spk_A", "audio_codes": [[...]]}
{"audio": "./data/spk_B/utt0001.wav", "text": "今天天气真不错，适合出去走走。", "ref_audio": "./data/spk_B/ref.wav", "speaker": "spk_B", "audio_codes": [[...]]}
```

**必需字段**：
- `audio`: 训练音频文件路径
- `text`: 文本转录
- `speaker`: 说话人 ID（用于分组）
- `audio_codes`: 音频编码
- `ref_audio`: 参考音频（训练时不使用，但需要包含）

### 步骤 2: 提取 Speaker Embeddings

使用新的提取脚本从训练数据中自动提取：

```bash
python extract_speaker_embeddings_from_jsonl.py \
    --model_path pretrained_models/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl train_data.jsonl \
    --output_path speaker_embeddings.pt
```

**工作原理**：
1. 读取 JSONL 文件
2. 按 `speaker` 字段分组
3. 对每个说话人，收集所有 `audio` 文件路径
4. 从这些 audio 文件提取 mel 并生成 speaker embeddings
5. 对每个说话人的所有 embeddings 取平均
6. 保存到 `.pt` 文件

输出示例：
```
Found 3 unique speakers:
  - spk_A: 150 audio files
  - spk_B: 120 audio files
  - spk_C: 80 audio files
```

### 步骤 3: 运行多说话人训练

使用新的训练脚本：

```bash
python sft_12hz_multi_speaker.py \
    --init_model_path pretrained_models/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl train_data.jsonl \
    --speaker_embeddings_path speaker_embeddings.pt \
    --batch_size 4 \
    --lr 2e-5 \
    --num_epochs 3 \
    --output_model_path output/multi_speaker_model
```

**关键变化**：
- ✅ 使用 `--speaker_embeddings_path` 指定预提取的 embeddings
- ✅ 移除了 `--speaker_name` 参数（不再需要）
- ✅ 支持批次中包含不同说话人

### 步骤 4: 使用微调后的模型

训练完成后，检查 `config.json` 查看说话人映射：

```json
{
  "talker_config": {
    "spk_id": {
      "spk_A": 3000,
      "spk_B": 3001,
      "spk_C": 3002
    }
  }
}
```

## 文件说明

### 原始文件（未修改）
- `dataset.py` - 原始数据集实现（单说话人）
- `sft_12hz.py` - 原始训练脚本（单说话人）

### 新增文件（多说话人）
- `extract_speaker_embeddings_from_jsonl.py` - 从训练 JSONL 提取 embeddings
- `dataset_multi_speaker.py` - 多说话人数据集（使用 `speaker` 字段）
- `sft_12hz_multi_speaker.py` - 多说话人训练脚本（加载预提取 embeddings）

## 与原实现的差异

### 原始方法（单说话人）
```python
# dataset.py
ref_audio_path = item['ref_audio']
ref_mel = self.extract_mels(audio=wav, sr=sr)
return {"ref_mel": ref_mel}

# sft_12hz.py
speaker_embedding = model.speaker_encoder(ref_mels).detach()
target_speaker_embedding = speaker_embedding  # 只保存第一个batch
```

### 新方法（多说话人）
```python
# dataset_multi_speaker.py
speaker = item['speaker']
return {"speaker": speaker}  # 只返回 ID

# sft_12hz_multi_speaker.py
speaker_embeddings_dict = torch.load(args.speaker_embeddings_path)
for speaker in batch['speakers']:
    speaker_embedding = speaker_embeddings_dict[speaker]  # 查表
```

## 优势

1. **多说话人支持**：同一批次可以包含不同说话人
2. **更好的 embedding 质量**：使用所有 audio 文件的平均 embedding
3. **训练效率**：不需要在训练时重复提取 mel 和 embedding
4. **灵活性**：可以轻松添加或更新说话人
5. **向后兼容**：原始文件保持不变，可以继续用于单说话人场景

## 注意事项

- 确保 JSONL 中的 `speaker` 字段一致（区分大小写）
- 预提取的 embeddings 应该在训练前完成
- 所有训练样本的 `speaker` 字段必须在 embeddings 文件中存在
- Speaker encoder 仍然是冻结的（不参与训练）
