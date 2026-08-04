"""Tokenizer utility module for text processing in Multi-Modal ViT."""

from typing import List, Dict, Union, Optional
import torch
from transformers import AutoTokenizer


class SimpleFallbackTokenizer:
    """Standalone fallback tokenizer when HuggingFace Hub is unreachable or offline."""

    def __init__(self, vocab_size: int = 30522, max_length: int = 64) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.unk_token_id = 100

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, Union[torch.Tensor, List[List[int]]]]:
        if isinstance(text, str):
            text = [text]

        length = max_length or self.max_length
        input_ids = []
        attn_masks = []

        for t in text:
            words = t.lower().split()
            ids = [self.cls_token_id] + [(abs(hash(w)) % (self.vocab_size - 200)) + 103 for w in words[: length - 2]] + [self.sep_token_id]
            pad_len = length - len(ids)
            if pad_len > 0:
                mask = [1] * len(ids) + [0] * pad_len
                ids = ids + [self.pad_token_id] * pad_len
            else:
                ids = ids[:length]
                mask = [1] * length

            input_ids.append(ids)
            attn_masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attn_masks}

    def decode(self, token_ids: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().squeeze().tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        words = [f"word_{tid}" for tid in token_ids if tid not in (0, 101, 102)]
        return " ".join(words) if words else "sample text caption"

    def batch_decode(self, batch_token_ids: Union[torch.Tensor, List[List[int]]], skip_special_tokens: bool = True) -> List[str]:
        if isinstance(batch_token_ids, torch.Tensor):
            batch_token_ids = batch_token_ids.cpu().tolist()
        return [self.decode(ids, skip_special_tokens) for ids in batch_token_ids]


class MultiModalTokenizer:
    """Wrapper around HuggingFace Tokenizers with standard interfaces for MultiModal models."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 64,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.is_fallback = False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"[Warning] Failed to load '{model_name}' tokenizer ({e}). Utilizing SimpleFallbackTokenizer.")
            self.is_fallback = True
            self.tokenizer = SimpleFallbackTokenizer(vocab_size=30522, max_length=max_length)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None else 0

    @property
    def cls_token_id(self) -> int:
        return self.tokenizer.cls_token_id if hasattr(self.tokenizer, "cls_token_id") and self.tokenizer.cls_token_id is not None else 101

    @property
    def sep_token_id(self) -> int:
        return self.tokenizer.sep_token_id if hasattr(self.tokenizer, "sep_token_id") and self.tokenizer.sep_token_id is not None else 102

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id if hasattr(self.tokenizer, "unk_token_id") and self.tokenizer.unk_token_id is not None else 100

    def encode(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            text = [text]

        if self.is_fallback:
            return self.tokenizer(text, max_length=self.max_length, return_tensors=return_tensors)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=return_tensors,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
    ) -> str:
        if self.is_fallback:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().squeeze().tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        batch_token_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        if self.is_fallback:
            return self.tokenizer.batch_decode(batch_token_ids, skip_special_tokens=skip_special_tokens)

        if isinstance(batch_token_ids, torch.Tensor):
            batch_token_ids = batch_token_ids.cpu().tolist()
        return self.tokenizer.batch_decode(batch_token_ids, skip_special_tokens=skip_special_tokens)
