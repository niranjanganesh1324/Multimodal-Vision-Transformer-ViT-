"""Tokenizer utility module for text processing in Multi-Modal ViT."""

from typing import List, Dict, Union, Optional
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


class MultiModalTokenizer:
    """Wrapper around HuggingFace Tokenizers with standard interfaces for MultiModal models."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 64,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            model_name: HuggingFace tokenizer model name or path.
            max_length: Maximum sequence length.
            padding: Padding strategy ('max_length', 'longest', or False).
            truncation: Whether to truncate sequences exceeding max_length.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        try:
            self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = AutoTokenizer.from_pretrained(
                model_name
            )
        except Exception as e:
            print(f"[Warning] Failed to load '{model_name}' tokenizer from HuggingFace ({e}). Falling back to bert-base-uncased.")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Return padding token ID."""
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    @property
    def cls_token_id(self) -> int:
        """Return CLS / BOS token ID."""
        return self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else 101

    @property
    def sep_token_id(self) -> int:
        """Return SEP / EOS token ID."""
        return self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 102

    @property
    def unk_token_id(self) -> int:
        """Return UNK token ID."""
        return self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 100

    def encode(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text into input_ids and attention_mask tensors.

        Args:
            text: Input string or list of strings.
            return_tensors: Format for returned tensors ('pt' for PyTorch).

        Returns:
            Dict containing 'input_ids' and 'attention_mask'.
        """
        if isinstance(text, str):
            text = [text]

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
        """Decode token IDs back into human readable text.

        Args:
            token_ids: Tensor or list of integer token IDs.
            skip_special_tokens: Whether to omit [PAD], [CLS], [SEP], etc.

        Returns:
            Decoded text string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().squeeze().tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        batch_token_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token IDs into text strings.

        Args:
            batch_token_ids: Tensor of shape (B, S) or list of token lists.
            skip_special_tokens: Whether to omit special tokens.

        Returns:
            List of decoded text strings.
        """
        if isinstance(batch_token_ids, torch.Tensor):
            batch_token_ids = batch_token_ids.cpu().tolist()
        return self.tokenizer.batch_decode(batch_token_ids, skip_special_tokens=skip_special_tokens)
