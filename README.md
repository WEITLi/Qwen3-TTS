# Qwen3-TTS

Multi-speaker fine-tuning support with emotion and paralinguistic tag capabilities.

## Features

- Multi-speaker supervised fine-tuning (SFT)
- Emotion and paralinguistic tag support
- Token dropout training strategy

## Quick Start

### Multi-Speaker Fine-Tuning

Fine-tune the model with multiple speakers:

```bash
bash run_sft_multi_spk.sh
```

### Emotion & Paralinguistic Training

Train the model with emotion and paralinguistic features using token dropout.

**Prerequisites:**
- Place pretrained models in `./pretrained_models/`

**Steps:**

1. Initialize token dropout configuration:
   ```bash
   bash quick_start_token_dropout.sh
   ```

2. Run multi-speaker training with dropout:
   ```bash
   bash run_sft_multi_spk_dropout.sh
   ```

## Infer
There are two ways to infer based on sft:
1. direct use spk embedding:
   ```bash
   python infer_jsonl.py
   ```

2. In-Context Learning via ref_audio & ref_text (unstable currently):
   ```bash
   python infer_jsonl_icl.py
   ```