"""Text Encoder module supporting BERT, RoBERTa, DistilBERT, and standalone Transformer architectures."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CustomTextTransformer(nn.Module):
    """Fallback / standalone Text Transformer implementation for synthetic testing & lightweight usage."""

    def __init__(
        self,
        vocab_size: int = 30522,
        max_position_embeddings: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, S = input_ids.shape
        position_ids = torch.arange(S, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(B, -1)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(words_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        # PyTorch TransformerEncoder src_key_padding_mask: True means ignore (pad)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        hidden_state = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)
        cls_token = hidden_state[:, 0]

        return {
            "last_hidden_state": hidden_state,
            "cls_token": cls_token,
        }


class TextEncoder(nn.Module):
    """Unified Text Encoder wrapping HuggingFace language models or Custom Text Transformer."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        vocab_size: int = 30522,
        hidden_dim: int = 768,
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.use_custom_fallback = False

        if not pretrained:
            self.use_custom_fallback = True
            self.backbone = CustomTextTransformer(vocab_size=vocab_size, hidden_dim=hidden_dim)
            backbone_dim = hidden_dim
        else:
            try:
                self.backbone = AutoModel.from_pretrained(model_name)
                backbone_dim = self.backbone.config.hidden_size if hasattr(self.backbone.config, "hidden_size") else hidden_dim
            except Exception as e:
                print(f"[Warning] Could not load text encoder '{model_name}' ({e}). Utilizing standalone CustomTextTransformer.")
                self.use_custom_fallback = True
                self.backbone = CustomTextTransformer(vocab_size=vocab_size, hidden_dim=hidden_dim)
                backbone_dim = hidden_dim

        if backbone_dim != hidden_dim and not self.use_custom_fallback:
            self.proj = nn.Linear(backbone_dim, hidden_dim)
        else:
            self.proj = nn.Identity()

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for text encoder.

        Args:
            input_ids: Tensor of shape (B, S).
            attention_mask: Tensor of shape (B, S).

        Returns:
            Dict containing:
                - 'last_hidden_state': (B, S, hidden_dim)
                - 'cls_token': (B, hidden_dim)
        """
        if self.use_custom_fallback:
            return self.backbone(input_ids, attention_mask=attention_mask)

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = self.proj(outputs.last_hidden_state)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_token = self.proj(outputs.pooler_output)
        else:
            cls_token = hidden_state[:, 0]

        return {
            "last_hidden_state": hidden_state,
            "cls_token": cls_token,
        }
