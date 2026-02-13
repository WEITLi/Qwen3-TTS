#!/usr/bin/env python3
# coding=utf-8
"""
带特殊 Token Dropout 的 SFT 训练脚本
在训练时随机 dropout 特殊 tokens，让模型学会在有/无特殊 tokens 的情况下都能工作
"""

import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset_multi_spk_dropout import SpecialTokenDropout, TTSDatasetWithTokenDropout
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ConstantLR


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, 
                       default="pretrained_models/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--speaker_embeddings_path", type=str, required=True,
                        help="Path to pre-extracted speaker embeddings .pt file")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Token Dropout 相关参数
    parser.add_argument("--token_dropout", type=float, default=0.5,
                       help="Special token dropout rate")
    parser.add_argument("--special_tokens", type=str, nargs='*', default=None,
                       help="List of special tokens to dropout (auto-detect if not specified)")
    
    # other train args
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=100)
    
    # Learning rate scheduler args
    parser.add_argument("--lr_scheduler", type=str, default="warmup_constant",
                       choices=["warmup_constant", "constant", "cosine", "cosine_with_warmup"],
                       help="Learning rate scheduler type")
    parser.add_argument("--cosine_min_lr", type=float, default=0.0,
                       help="Minimum learning rate for cosine scheduler (as ratio of initial lr)")
    
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16"
    )

    MODEL_PATH = args.init_model_path

    # load model and tokenizer
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Init Token Dropout
    token_dropout = None
    if args.token_dropout > 0:
        token_dropout = SpecialTokenDropout(
            tokenizer,
            dropout_rate=args.token_dropout,
            special_token_patterns=args.special_tokens
        )
    # Load pre-extracted speaker embeddings
    print(f"Loading speaker embeddings from {args.speaker_embeddings_path}...")
    speaker_embeddings_dict = torch.load(args.speaker_embeddings_path)
    print(f"Loaded embeddings for {len(speaker_embeddings_dict)} speakers:")
    for speaker in speaker_embeddings_dict.keys():
        print(f"  - {speaker}")

    # prepare data
    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    
    dataset = TTSDatasetWithTokenDropout(
        train_data, 
        qwen3tts.processor, 
        config, 
        lag_num=-1,
        token_dropout=token_dropout
    )
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate_fn
    )
   # optimizer
    optimizer = AdamW(
        qwen3tts.model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    # Calculate total training steps for cosine scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # lr_scheduler
    if args.lr_scheduler == "warmup_constant":
        # Original: warmup then constant
        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                return float(current_step) / float(max(1, args.warmup_steps))
            return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif args.lr_scheduler == "constant":
        # Pure constant learning rate (no warmup)
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)
        
    elif args.lr_scheduler == "cosine":
        # Pure cosine annealing (no warmup)
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=args.lr * args.cosine_min_lr
        )
        
    elif args.lr_scheduler == "cosine_with_warmup":
        # Warmup then cosine annealing
        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                # Warmup phase
                return float(current_step) / float(max(1, args.warmup_steps))
            else:
                # Cosine annealing phase
                progress = (current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
                return args.cosine_min_lr + (1.0 - args.cosine_min_lr) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
        scheduler = LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")

    # Accelerator 准备
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, scheduler
    )

    # 用于保存每个 speaker ID 的 embedding
    learned_speaker_embeddings = {}

    num_epochs = args.num_epochs
    global_step = 0
    
    accelerator.print("=" * 60)
    accelerator.print("Training Configuration:")
    accelerator.print(f"  Model: {MODEL_PATH}")
    accelerator.print(f"  Output: {args.output_model_path}")
    accelerator.print(f"  Epochs: {num_epochs}")
    accelerator.print(f"  Batch size: {args.batch_size}")
    accelerator.print(f"  Learning rate: {args.lr}")
    accelerator.print(f"  LR Scheduler: {args.lr_scheduler}")
    if args.lr_scheduler in ["cosine", "cosine_with_warmup"]:
        accelerator.print(f"  Cosine min LR ratio: {args.cosine_min_lr}")
    if args.lr_scheduler in ["warmup_constant", "cosine_with_warmup"]:
        accelerator.print(f"  Warmup steps: {args.warmup_steps}")
    accelerator.print(f"  Token dropout: {args.token_dropout}")
    if token_dropout:
        accelerator.print(f"  Special tokens: {token_dropout.special_tokens}")
    accelerator.print(f"  Weight decay: {args.weight_decay}")
    accelerator.print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total training samples: {len(train_data)}")
    accelerator.print(f"  Total training steps: {total_steps}")
    accelerator.print("=" * 60)

    model.train()
    
    # train loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # 准备输入
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']
                speakers = batch['speakers']

                speaker_embedding_list = []
                for speaker in speakers:
                    if speaker not in speaker_embeddings_dict:
                        raise ValueError(f"Speaker '{speaker}' not found in speaker embeddings file. "
                                       f"Available speakers: {list(speaker_embeddings_dict.keys())}")
                    spk_emb = speaker_embeddings_dict[speaker].to(model.device).to(model.dtype)
                    speaker_embedding_list.append(spk_emb)
                speaker_embedding = torch.cat(speaker_embedding_list, dim=0)

                # 准备输入 embeddings
                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                
                input_codec_embedding[:, 6, :] = speaker_embedding
                input_embeddings = input_text_embedding + input_codec_embedding

                # 添加 codec embeddings
                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # Forward pass
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                # 计算 sub-talker loss
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + sub_talker_loss*0.3
                
                # Backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

            # Logging
            if global_step % args.logging_steps == 0:
                
                current_lr = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} |"  f"main_Loss: {outputs.loss.item():.4f} | "  f"sub_Loss: {sub_talker_loss.item():.4f} |"
                    f"LR: {current_lr:.2e}"
                )
            
        # Epoch-save ckpt
        
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "base"
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

            # Keep speaker_encoder in state_dict for ICL support
            # If you want to use ICL (In-Context Learning) with reference audio,
            # you MUST keep the speaker_encoder in the checkpoint.
            # Uncomment the lines below ONLY if you don't need ICL:
            # drop_prefix = "speaker_encoder"
            # keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            # for k in keys_to_drop:
            #     del state_dict[k]

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
