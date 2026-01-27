# 多说话人微调快速开始

## 数据准备

将您的训练数据准备为 JSONL 格式（每行一个样本）：

```json
{"audio":"./data/spk_A/utt0001.wav","text":"转录文本...","ref_audio":"./data/spk_A/ref.wav","speaker":"spk_A"}
{"audio":"./data/spk_B/utt0001.wav","text":"其他文本...","ref_audio":"./data/spk_B/ref.wav","speaker":"spk_B"}
```

**必需字段**：
- `audio`: 训练音频路径
- `text`: 文本转录  
- `speaker`: 说话人 ID（用于分组和平均）
- `ref_audio`: 参考音频（保留但不使用）

## 一键运行

```bash
# 编辑配置
vim run_sft_multi_spk.sh  # 修改数据路径和参数

# 运行完整流程
bash run_sft_multi_spk.sh
```

脚本会自动执行：
1. **数据准备**：提取 audio codes
2. **Embedding 提取**：对每个说话人的所有 audio 提取并平均 speaker embeddings
3. **多说话人训练**：使用预提取的 embeddings 进行微调

## 分步运行

如果需要单独运行每一步：

### 步骤 1: 准备训练数据
```bash
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path ./pretrained_models/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ./data/train_raw.jsonl \
  --output_jsonl ./data/train_with_codes.jsonl
```

### 步骤 2: 提取 speaker embeddings
```bash
python finetuning/extract_speaker_embeddings_from_jsonl.py \
  --model_path ./pretrained_models/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl ./data/train_with_codes.jsonl \
  --output_path ./data/speaker_embeddings.pt \
  --device cuda:0
```

**重要**：此步骤会读取 JSONL，按 `speaker` 字段分组，使用每个说话人的**所有 audio 文件**提取 mel spectrograms，再从 mel 提取 speaker embeddings，最后取平均。

输出示例：
```
Found 3 unique speakers:
  - spk_A: 150 audio files
  - spk_B: 120 audio files
  - spk_C: 80 audio files

[1/3] Processing speaker: spk_A
  Audio files: 150
  → Processed 150 audio files, averaged embedding shape: torch.Size([1, 256])
```

### 步骤 3: 多说话人训练
```bash
python finetuning/sft_12hz_multi_speaker.py \
  --init_model_path ./pretrained_models/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl ./data/train_with_codes.jsonl \
  --speaker_embeddings_path ./data/speaker_embeddings.pt \
  --batch_size 4 \
  --lr 2e-5 \
  --num_epochs 3 \
  --output_model_path ./output/multi_speaker_model
```

## 查看结果

训练完成后，检查输出目录中的 `config.json`：

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

每个说话人都被分配了一个唯一的 ID，可以在推理时使用。

## 关键特性

✅ **平均 embeddings**：对每个说话人的所有音频文件提取 embedding 并取平均，得到更稳定的说话人表征
✅ **批次灵活**：同一批次可以包含不同说话人的样本
✅ **高效训练**：embeddings 只提取一次，训练时直接查表
✅ **保留原文件**：`dataset.py` 和 `sft_12hz.py` 原文件未修改

## 文件说明

- `extract_speaker_embeddings_from_jsonl.py` - 从 JSONL 提取并平均 embeddings
- `dataset_multi_speaker.py` - 多说话人数据集（返回 speaker ID）
- `sft_12hz_multi_speaker.py` - 多说话人训练脚本（加载预提取 embeddings）
- `run_sft_multi_spk.sh` - 一键运行脚本

更多详情见 `MULTI_SPEAKER_GUIDE_CN.md`
