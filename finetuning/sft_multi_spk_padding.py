# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset_multi_spk_padding import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="..pretrained_models/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    # parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config, start_id=2861, max_id=3066)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )
    # [新增] 用于保存每个 ID 最新计算出的 Embedding，用于最后写入权重
    # key: int (spk_id), value: torch.Tensor
    learned_speaker_embeddings = {}

    num_epochs = args.num_epochs
    model.train()
    # train loops and data load
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']
                spk_ids = batch['spk_ids'] # [新增] 从 batch 获取 IDs

                # spk emb: use 【ref_mels】 音频特征-> spk meb
                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype))
                spk_emb_detached = speaker_embedding.detach()
                # if target_speaker_embedding is None:
                #     target_speaker_embedding = speaker_embedding
                for i, spk_id in enumerate(spk_ids):
                    # 获取当前样本的 ID (int)
                    sid = spk_id.item()
                    # 获取当前样本的 embedding (1, hidden_dim) -> (hidden_dim)
                    semb = spk_emb_detached[i].squeeze(0).cpu() 
                    learned_speaker_embeddings[sid] = semb

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                
                # input_codec_embedding[:, 6, :] = speaker_embedding
                input_codec_embedding[:, 6, :] = speaker_embedding.squeeze(1)

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding
                # 4. forward and compute loss
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + sub_talker_loss
                # 5. update params
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
        # 6. save ckpts
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            # [New]
            talker_config = config_dict.get("talker_config", {})

            # 构建 {speaker_name: id} 写入 config
            # 同时构建 {speaker_name: is_dialect} (默认为 False)
            spk_id_map = {}
            spk_dialect_map = {}
            
            # 如果是 DDP，dataset 在其他进程可能不同步，但 mapping 逻辑是一致的 (只要数据顺序一致)
            # 或者通过 accelerator.gather 来收集，但这里简化处理，假设数据完全一致
            for name, sid in dataset.speaker_map.items():
                spk_id_map[name] = sid
                spk_dialect_map[name] = False   # [尚未收集]
                
            talker_config["spk_id"] = spk_id_map
            talker_config["spk_is_dialect"] = spk_dialect_map
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            # 确保 weight 存在且在 CPU 上 (上面已经 detach 到 cpu 了)
            codec_embedding_weight = state_dict['talker.model.codec_embedding.weight']
            
            print(f"Saving embeddings for {len(learned_speaker_embeddings)} speakers...")
            
            for spk_id, emb_tensor in learned_speaker_embeddings.items():
                if 2861 <= spk_id <= 3066: # 双重保险检查范围
                    # 注意类型转换，保持和权重一致 (通常是 bf16 或 fp32)
                    codec_embedding_weight[spk_id] = emb_tensor.to(codec_embedding_weight.dtype)
                else:
                    print(f"Skipping ID {spk_id} out of range [2861, 3066]")

            # 覆盖回去
            state_dict['talker.model.codec_embedding.weight'] = codec_embedding_weight
            
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            accelerator.print(f"Model saved to {save_path} with updated speaker embeddings.")

if __name__ == "__main__":
    train()
