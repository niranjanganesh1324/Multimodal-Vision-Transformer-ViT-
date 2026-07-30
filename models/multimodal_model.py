"""Unified Multi-Modal Vision Transformer Architecture for Joint Contrastive, Captioning, and VQA."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.cross_attention import CrossModalFusionDecoder


class MultiModalViT(nn.Module):
    """Unified Multi-Modal Vision Transformer model integrating contrastive alignment,
    cross-attention fusion, image captioning generation, and VQA prediction.
    """

    def __init__(
        self,
        vision_model_name: str = "google/vit-base-patch16-224",
        text_model_name: str = "bert-base-uncased",
        hidden_dim: int = 768,
        projection_dim: int = 512,
        num_fusion_layers: int = 4,
        num_heads: int = 8,
        vocab_size: int = 30522,
        vqa_num_classes: int = 3129,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.vocab_size = vocab_size
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Encoders
        self.vision_encoder = VisionEncoder(model_name=vision_model_name, hidden_dim=hidden_dim)
        self.text_encoder = TextEncoder(model_name=text_model_name, vocab_size=vocab_size, hidden_dim=hidden_dim)

        # Contrastive Projection Heads (Normalized Joint Embedding Space)
        self.vision_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Cross-Modal Fusion Decoder
        self.fusion_decoder = CrossModalFusionDecoder(
            num_layers=num_fusion_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Task Heads
        # 1. Causal LM Caption Head (Predicts vocabulary logits for text generation)
        self.caption_head = nn.Linear(hidden_dim, vocab_size)

        # 2. VQA Answer Head (Classifies answers over vqa_num_classes)
        self.vqa_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vqa_num_classes),
        )

        # 3. Image-Text Matching (ITM) Binary Classification Head
        self.itm_head = nn.Linear(hidden_dim, 2)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create lower-triangular causal self-attention mask (seq_len, seq_len)."""
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask

    def encode_image(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract visual patch embeddings and projected contrastive features."""
        vision_out = self.vision_encoder(pixel_values)
        cls_feat = vision_out["cls_token"]
        projected_feat = F.normalize(self.vision_proj(cls_feat), p=2, dim=-1)
        vision_out["projected_embeds"] = projected_feat
        return vision_out

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract textual sequence embeddings and projected contrastive features."""
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = text_out["cls_token"]
        projected_feat = F.normalize(self.text_proj(cls_feat), p=2, dim=-1)
        text_out["projected_embeds"] = projected_feat
        return text_out

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = "multitask",
    ) -> Dict[str, torch.Tensor]:
        """Unified multi-modal forward pass.

        Args:
            pixel_values: Tensor of shape (B, C, H, W).
            input_ids: Tensor of shape (B, S).
            attention_mask: Tensor of shape (B, S).
            task: Execution task mode ('contrastive', 'caption', 'vqa', 'multitask').

        Returns:
            Dict containing relevant predictions, logits, embeddings, and attention maps.
        """
        device = pixel_values.device
        results = {}

        # 1. Encode Image and Text
        vision_out = self.encode_image(pixel_values)
        text_out = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)

        results["image_embeds"] = vision_out["projected_embeds"]
        results["text_embeds"] = text_out["projected_embeds"]
        results["temperature"] = torch.clamp(self.temperature, min=0.01, max=1.0)

        if task == "contrastive":
            return results

        # 2. Cross-Modal Fusion
        causal_mask = self.generate_causal_mask(input_ids.shape[1], device=device)
        fused_states, cross_attn_map = self.fusion_decoder(
            text_embeds=text_out["last_hidden_state"],
            image_embeds=vision_out["patch_tokens"],
            text_causal_mask=causal_mask,
        )

        results["fused_hidden_states"] = fused_states
        results["cross_attention_map"] = cross_attn_map

        # 3. Compute Task Logits
        # Image Captioning logits: (B, S, Vocab)
        caption_logits = self.caption_head(fused_states)
        results["caption_logits"] = caption_logits

        # VQA Logits: (B, VQA_Classes) using the fused [CLS] representation
        vqa_cls = fused_states[:, 0]
        vqa_logits = self.vqa_head(vqa_cls)
        results["vqa_logits"] = vqa_logits

        # ITM Logits: (B, 2)
        itm_logits = self.itm_head(vqa_cls)
        results["itm_logits"] = itm_logits

        return results
