"""Vision Encoder module supporting ViT, Swin, DINOv2, and OpenCLIP architectures."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CustomViTBackbone(nn.Module):
    """Fallback / lightweight Vision Transformer implementation for synthetic testing & standalone usage."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_dim = hidden_dim

        # Patch Embedding projection
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = pixel_values.shape[0]
        # (B, C, H, W) -> (B, D, H_p, W_p) -> (B, N_patches, D)
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        cls_token_out = x[:, 0]
        patch_tokens_out = x[:, 1:]
        return {
            "last_hidden_state": x,
            "cls_token": cls_token_out,
            "patch_tokens": patch_tokens_out,
        }


class VisionEncoder(nn.Module):
    """Unified Vision Encoder wrapping pretrained Transformers (ViT, Swin, DINOv2) or Custom ViT."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
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
            self.backbone = CustomViTBackbone(hidden_dim=hidden_dim)
            backbone_dim = hidden_dim
        else:
            try:
                self.backbone = AutoModel.from_pretrained(model_name)
                if hasattr(self.backbone.config, "hidden_size"):
                    backbone_dim = self.backbone.config.hidden_size
                else:
                    backbone_dim = hidden_dim
            except Exception as e:
                print(f"[Warning] Could not load vision encoder checkpoint '{model_name}' ({e}). Utilizing standalone CustomViTBackbone.")
                self.use_custom_fallback = True
                self.backbone = CustomViTBackbone(hidden_dim=hidden_dim)
                backbone_dim = hidden_dim

        # Projection layer to match target hidden_dim if backbone dimension differs
        if backbone_dim != hidden_dim and not self.use_custom_fallback:
            self.proj = nn.Linear(backbone_dim, hidden_dim)
        else:
            self.proj = nn.Identity()

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for vision encoder.

        Args:
            pixel_values: Tensor of shape (B, C, H, W).

        Returns:
            Dict containing:
                - 'cls_token': (B, hidden_dim)
                - 'patch_tokens': (B, N_patches, hidden_dim)
                - 'last_hidden_state': (B, 1 + N_patches, hidden_dim)
        """
        if self.use_custom_fallback:
            return self.backbone(pixel_values)

        outputs = self.backbone(pixel_values=pixel_values)

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_state = self.proj(outputs.last_hidden_state)
            cls_token = hidden_state[:, 0]
            patch_tokens = hidden_state[:, 1:]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_token = self.proj(outputs.pooler_output)
            patch_tokens = self.proj(outputs.last_hidden_state[:, 1:]) if hasattr(outputs, "last_hidden_state") else cls_token.unsqueeze(1)
            hidden_state = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        else:
            cls_token = outputs[0][:, 0]
            cls_token = self.proj(cls_token)
            patch_tokens = self.proj(outputs[0][:, 1:])
            hidden_state = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)

        return {
            "last_hidden_state": hidden_state,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }
