# 1.prepare_data.py - 数据预处理脚本
## 功能
该脚本主要负责将原始音频数据转换为模型可接受的格式，具体包括：

### 音频编码：
使用 Qwen3-TTS 专用的音频编码器将音频文件转换为离散音频编码（Audio Codes）
### 数据格式化：
将处理后的数据保存为 JSONL 格式，用于后续的模型训练
输入数据格式
{
  "audio": "path/to/audio.wav",
  "text": "需要合成的文本"
}

输出数据格式
{
  "audio": "path/to/audio.wav",
  "text": "需要合成的文本",
  "audio_codes": [123, 456, 789, ...]  // 新增音频编码字段
}
# 2.sft_12hz.py - 模型微调脚本
## 2.1功能
该脚本用于对 Qwen3-TTS 模型进行说话人自适应微调（SFT），生成特定说话人的自定义语音模型。
## 2.2 核心流程
### 2.2.1 训练过程特点
梯度累积：使用 gradient_accumulation_steps=4 实现大批次训练
混合精度训练：使用 bf16 精度加速训练
注意力优化：使用 flash_attention_2 提升注意力计算效率
说话人嵌入优化：专门优化目标说话人的嵌入向量
### 2.2.2 关键参数
--init_model_path：预训练模型路径（默认：Qwen/Qwen3-TTS-12Hz-1.7B-Base）
--train_jsonl：训练数据路径（由 prepare_data.py 生成）
--output_model_path：输出模型保存路径
--speaker_name：目标说话人名称（用于标识自定义语音）
### 2.2.3 输出内容
训练检查点：每个 epoch 保存一个检查点
自定义模型：最终生成包含目标说话人信息的完整模型
配置更新：自动修改模型配置，添加说话人信息
## 2.3 微调流程
这两个脚本通常配合使用，形成完整的 TTS 微调 pipeline：

数据准备：运行 prepare_data.py 处理原始音频数据
模型微调：运行 sft_12hz.py 对模型进行说话人自适应训练
模型使用：使用生成的自定义模型进行 TTS 合成