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
from dataset_multi_spk import TTSDataset  # Use multi-speaker dataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

speaker_embeddings_dict = None  # Global dict to store pre-extracted speaker embeddings

def train():
    global speaker_embeddings_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="..pretrained_models/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--speaker_embeddings_path", type=str, required=True,
                        help="Path to pre-extracted speaker embeddings .pt file")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()

    # logging_dir = os.path.join(args.output_model_path, "logs")

    accelerator = Accelerator(gradient_accumulation_steps=1, 
                              mixed_precision="bf16", 
                              )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Load pre-extracted speaker embeddings
    print(f"Loading speaker embeddings from {args.speaker_embeddings_path}...")
    speaker_embeddings_dict = torch.load(args.speaker_embeddings_path)
    print(f"Loaded embeddings for {len(speaker_embeddings_dict)} speakers:")
    for speaker in speaker_embeddings_dict.keys():
        print(f"  - {speaker}")

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()
    # train loops and data load
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']
                speakers = batch['speakers']  # List of speaker IDs
                
                # Get speaker embeddings for each sample in batch from pre-extracted dict
                speaker_embedding_list = []
                for speaker in speakers:
                    if speaker not in speaker_embeddings_dict:
                        raise ValueError(f"Speaker '{speaker}' not found in speaker embeddings file. "
                                       f"Available speakers: {list(speaker_embeddings_dict.keys())}")
                    spk_emb = speaker_embeddings_dict[speaker].to(model.device).to(model.dtype)
                    speaker_embedding_list.append(spk_emb)
                speaker_embedding = torch.cat(speaker_embedding_list, dim=0)  # [batch_size, emb_dim]

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

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
            talker_config = config_dict.get("talker_config", {})
            
            # Register all speakers in config
            spk_id_mapping = {}
            for idx, spk_name in enumerate(sorted(speaker_embeddings_dict.keys())):
                spk_id_mapping[spk_name] = 3000 + idx
            
            talker_config["spk_id"] = spk_id_mapping
            talker_config["spk_is_dialect"] = {spk: False for spk in spk_id_mapping.keys()}
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            # Remove speaker_encoder from state_dict (not needed for inference)
            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            # Build speaker ID mapping (starting from index 3000)
            spk_id_mapping = {}
            for idx, spk_name in enumerate(sorted(speaker_embeddings_dict.keys())):
                spk_id_mapping[spk_name] = 3000 + idx
            
            # Extend codec_embedding weight to accommodate all speakers
            weight = state_dict['talker.model.codec_embedding.weight']
            max_speaker_idx = max(spk_id_mapping.values())
            if max_speaker_idx >= weight.shape[0]:
                # Need to expand the embedding weight
                new_size = max_speaker_idx + 1
                new_weight = torch.zeros(new_size, weight.shape[1], dtype=weight.dtype, device=weight.device)
                new_weight[:weight.shape[0]] = weight
                weight = new_weight
            
            # Inject all speaker embeddings into codec_embedding weight
            for spk_name, spk_idx in spk_id_mapping.items():
                weight[spk_idx] = speaker_embeddings_dict[spk_name][0].detach().to(weight.device).to(weight.dtype)
            
            state_dict['talker.model.codec_embedding.weight'] = weight
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()