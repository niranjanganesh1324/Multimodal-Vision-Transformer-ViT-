"""Cross-Attention and Transformer Decoder Blocks for Cross-Modal Vision-Language Fusion."""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross Attention layer with attention map extraction capabilities."""

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query: Tensor of shape (B, S_q, D) - Target sequence attending.
            key: Tensor of shape (B, S_k, D) - Source sequence being attended to.
            value: Tensor of shape (B, S_k, D) - Source sequence values.
            key_padding_mask: (B, S_k) boolean mask where True indicates padding.
            attn_mask: (S_q, S_k) causal or custom attention mask.

        Returns:
            Tuple of:
                - Output tensor of shape (B, S_q, D)
                - Attention weight matrix of shape (B, num_heads, S_q, S_k)
        """
        B, S_q, D = query.shape
        S_k = key.shape[1]

        Q = self.q_proj(query).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S_q, d_h)
        K = self.k_proj(key).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)      # (B, H, S_k, d_h)
        V = self.v_proj(value).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, S_k, d_h)

        # Scaled Dot-Product Attention: (B, H, S_q, S_k)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            # key_padding_mask shape (B, S_k) -> reshape to (B, 1, 1, S_k)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # Compute context: (B, H, S_q, d_h)
        context = torch.matmul(attn_weights_drop, V)
        # Reshape to (B, S_q, D)
        context = context.transpose(1, 2).contiguous().view(B, S_q, D)

        output = self.out_proj(context)
        return output, attn_weights


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block featuring Self-Attention, Cross-Attention, and Feed-Forward Network."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)

        self.norm3 = nn.LayerNorm(hidden_dim)
        act_layer = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Target query sequence tensor (B, S_q, D).
            encoder_hidden_states: Source context sequence tensor (B, S_k, D).
            self_attn_mask: Causal or custom self-attention mask (S_q, S_q).
            encoder_padding_mask: Key padding mask for cross-attention (B, S_k).

        Returns:
            Tuple of (output tensor (B, S_q, D), cross_attention_map (B, H, S_q, S_k))
        """
        # 1. Self Attention with Residual connection
        norm_x = self.norm1(x)
        self_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=self_attn_mask)
        x = x + self_out

        # 2. Cross Attention with Residual connection
        norm_x = self.norm2(x)
        cross_out, cross_attn_weights = self.cross_attn(
            query=norm_x,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=encoder_padding_mask,
        )
        x = x + cross_out

        # 3. Feed Forward Network with Residual connection
        x = x + self.ffn(self.norm3(x))

        return x, cross_attn_weights


class CrossModalFusionDecoder(nn.Module):
    """Stack of Transformer Decoder blocks for joint multimodal representation learning."""

    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 768,
        num_heads: int = 8,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        text_causal_mask: Optional[torch.Tensor] = None,
        image_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse text queries with image patch tokens via cross-attention layers.

        Args:
            text_embeds: (B, S_t, D)
            image_embeds: (B, S_i, D)
            text_causal_mask: (S_t, S_t)
            image_padding_mask: (B, S_i)

        Returns:
            Tuple of (fused_hidden_states, last_layer_cross_attention_map)
        """
        x = text_embeds
        last_attn = None

        for layer in self.layers:
            x, last_attn = layer(
                x=x,
                encoder_hidden_states=image_embeds,
                self_attn_mask=text_causal_mask,
                encoder_padding_mask=image_padding_mask,
            )

        x = self.norm(x)
        return x, last_attn
