# coding=utf-8
from typing import Any, List, Tuple, Union

import random
import re
import librosa
import numpy as np
import torch
from dataset_multi_spk import TTSDataset
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]

class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config:Qwen3TTSConfig, lag_num = -1):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config

    def __len__(self):
        return len(self.data_list)
    
    def _norm_wav_scale(self, wav: np.ndarray, scale: float = 0.7) -> np.ndarray:
        """
        Normalize audio waveform to a target peak amplitude to prevent clipping and ensure consistent volume.
        
        Args:
            wav:
                Input audio waveform as numpy array.
            scale:
                Target peak amplitude for normalization (default: 0.7 to avoid clipping).
        
        Returns:
            Normalized audio waveform as numpy array.
        """
        max_peak = np.max(np.abs(wav))
        if max_peak > 1e-6:
            gain = scale / max_peak
            norm_wav = wav * gain
        else:
            norm_wav = wav
        return norm_wav
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        
        audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        # new: Resample
        target_sr = 24000
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        # new: norm
        audio = self._norm_wav_scale(audio)

        return audio.astype(np.float32), target_sr

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - np.ndarray: waveform (NOT allowed alone here because sr is unknown)
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    
    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        ).transpose(1, 2)
        return mels



    def __getitem__(self, idx):
        item = self.data_list[idx]

        audio_path  = item["path"]
        text        = item["text"]
        audio_codes = item["audio_codes"]
        language    = item.get('language','Auto')
        speaker     = item['speaker']  # Use 'speaker' field for multi-speaker training

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        return {
            "text_ids": text_ids[:,:-5],    # 1 , t
            "audio_codes":audio_codes,      # t, 16
            "speaker":speaker               # Return speaker ID for pre-extracted embedding lookup
        }
        
    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b,t = len(batch),max_length

        input_ids   = torch.zeros((b,t,2),dtype=torch.long)
        codec_ids   = torch.zeros((b,t,16),dtype=torch.long)
        text_embedding_mask     = torch.zeros((b,t),dtype=torch.bool)
        codec_embedding_mask    = torch.zeros((b,t),dtype=torch.bool)
        codec_mask      = torch.zeros((b,t),dtype=torch.bool)
        attention_mask  = torch.zeros((b,t),dtype=torch.long)
        codec_0_labels  = torch.full((b, t), -100, dtype=torch.long)

        for i,data in enumerate(batch):
            text_ids        = data['text_ids']
            audio_codec_0   = data['audio_codes'][:,0]
            audio_codecs    = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]
            
            # text channel
            input_ids[i,  :3, 0] = text_ids[0,:3]   # <|im_start|>assistant\n : 这3个
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i,   7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0,3:]    # {text}<|im_end|>\n<|im_start|>assistant\n
            input_ids[i,   8+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len , 0] = self.config.tts_pad_token_id
            text_embedding_mask[i,  :8+text_ids_len+codec_ids_len] = True

            # codec channel
            # input_ids[i,   :3, 1] = 0
            input_ids[i,    3:8 ,1] = torch.tensor(
                                        [
                                            self.config.talker_config.codec_nothink_id,
                                            self.config.talker_config.codec_think_bos_id,
                                            self.config.talker_config.codec_think_eos_id,
                                            0,     # for speaker embedding
                                            self.config.talker_config.codec_pad_id       
                                        ]
                                    )
            input_ids[i,    8:8+text_ids_len-3  ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-3    ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-2    ,1] = self.config.talker_config.codec_bos_id
            input_ids[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,    1] = audio_codec_0
            input_ids[i,    8+text_ids_len-1+codec_ids_len,    1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i,    8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,:] = audio_codecs

            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False       # for speaker embedding

            codec_mask[i,   8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True
        
        # Collect speaker IDs for each sample in the batch
        speakers = [data['speaker'] for data in batch]

        return {
            'input_ids':input_ids,
            'speakers':speakers,  # Return speaker IDs for pre-extracted embedding lookup
            'attention_mask':attention_mask,
            'text_embedding_mask':text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask':codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels':codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask':codec_mask
        }



class SpecialTokenDropout:
    """特殊 Token Dropout 处理器
    
    在训练时随机 dropout 特殊 tokens，让模型学会在有/无特殊 tokens 的情况下都能工作
    """
    
    def __init__(self, tokenizer, dropout_rate=0.5, special_token_patterns=None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            dropout_rate: dropout 概率 (0.0-1.0)
            special_token_patterns: 要 dropout 的特殊 token 模式列表
                                   例如: ['[laughter]', '<strong>', '</strong>']
        """
        self.tokenizer = tokenizer
        self.dropout_rate = dropout_rate
        
        # 如果没有指定，自动检测非标准的特殊 tokens
        if special_token_patterns is None:
            self.special_tokens = self._detect_custom_tokens()
        else:
            self.special_tokens = special_token_patterns
        
        print(f"Special Token Dropout initialized:")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Special tokens to dropout: {self.special_tokens}")
    
    def _detect_custom_tokens(self):
        """自动检测自定义的特殊 tokens"""
        # 标准的 Qwen3-TTS tokens（不被 dropout）
        standard_tokens = {
            '<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>',
            '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>',
            '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>',
            '<|image_pad|>', '<|video_pad|>', '<|audio_start|>', '<|audio_end|>',
            '<tts_pad>', '<tts_text_bos>', '<tts_text_bos_single>', '<|audio_pad|>',
            '<|endoftext|>', '<tool_call>', '</tool_call>', '<|fim_prefix|>',
            '<|fim_middle|>', '<|fim_suffix|>', '<|fim_pad|>', '<|repo_name|>',
            '<|file_sep|>', '<tool_response>', '</tool_response>', '<think>', '</think>',
            '<tts_text_eod>'
        }
        
        # 获取所有特殊 tokens
        all_special = set(self.tokenizer.additional_special_tokens)
        
        # 自定义 tokens = 所有特殊 tokens - 标准 tokens
        custom_tokens = list(all_special - standard_tokens)
        
        return custom_tokens
    
    def apply_dropout(self, text):
        """
        对文本中的特殊 tokens 应用 dropout
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        # if random.random() > self.dropout_rate:
        #     # 不进行 dropout
        #     return text
        
        # 对每个特殊 token 进行 dropout
        result = text
        for token in self.special_tokens:
            
            escaped_token = re.escape(token)
            
            matches = list(re.finditer(escaped_token, result))
            
            # rate del every token
            for match in reversed(matches):  # back to front-avoicd index issue
                if random.random() < self.dropout_rate:
                    # del this token
                    start, end = match.span()
                    result = result[:start] + result[end:]
        
        return result


class TTSDatasetWithTokenDropout(TTSDataset):

    
    def __init__(self, data_list, processor, config, lag_num=-1, token_dropout=None):
        """
        Args:
            data_list: 训练数据列表
            processor: TTS processor
            config: 模型配置
            lag_num: lag number (default: -1)
            token_dropout: SpecialTokenDropout 实例，如果为 None 则不应用 dropout
        """
        super().__init__(data_list, processor, config, lag_num)
        self.token_dropout = token_dropout
    
    def __getitem__(self, idx):
        """获取一个样本，应用 token dropout"""
        item = self.data_list[idx]
        
        # 应用 token dropout
        if self.token_dropout is not None and 'rich_text' in item:
            # 如果有 rich_text 字段，对其应用 dropout 后赋值给 text
            original_text = item['rich_text']
            item['text'] = self.token_dropout.apply_dropout(original_text)
        # elif self.token_dropout is not None and 'text' in item:
        #     # 如果只有 text 字段，直接对其应用 dropout
        #     item['text'] = self.token_dropout.apply_dropout(item['text'])
        
        # 调用父类的处理逻辑
        return super().__getitem__(idx)
