"""Unit tests for Vision Encoder, Text Encoder, Cross-Attention, and MultiModalViT."""

import pytest
import torch

from models.vision_encoder import VisionEncoder, CustomViTBackbone
from models.text_encoder import TextEncoder, CustomTextTransformer
from models.cross_attention import MultiHeadCrossAttention, TransformerDecoderBlock, CrossModalFusionDecoder
from models.multimodal_model import MultiModalViT


def test_custom_vit_backbone():
    x = torch.randn(2, 3, 224, 224)
    model = CustomViTBackbone(image_size=224, patch_size=16, hidden_dim=256)
    out = model(x)

    assert "cls_token" in out
    assert "patch_tokens" in out
    assert out["cls_token"].shape == (2, 256)
    assert out["patch_tokens"].shape == (2, 196, 256)


def test_custom_text_transformer():
    input_ids = torch.randint(0, 1000, (2, 32))
    attn_mask = torch.ones(2, 32)
    model = CustomTextTransformer(vocab_size=1000, hidden_dim=256)
    out = model(input_ids, attention_mask=attn_mask)

    assert "cls_token" in out
    assert "last_hidden_state" in out
    assert out["cls_token"].shape == (2, 256)
    assert out["last_hidden_state"].shape == (2, 32, 256)


def test_vision_encoder():
    x = torch.randn(2, 3, 224, 224)
    encoder = VisionEncoder(model_name="google/vit-base-patch16-224", hidden_dim=512, pretrained=False)
    out = encoder(x)

    assert out["cls_token"].shape == (2, 512)
    assert out["patch_tokens"].shape[0] == 2


def test_text_encoder():
    input_ids = torch.randint(0, 1000, (2, 16))
    attn_mask = torch.ones(2, 16)
    encoder = TextEncoder(model_name="bert-base-uncased", hidden_dim=512, pretrained=False)
    out = encoder(input_ids, attention_mask=attn_mask)

    assert out["cls_token"].shape == (2, 512)
    assert out["last_hidden_state"].shape == (2, 16, 512)


def test_cross_attention_decoder():
    text_embeds = torch.randn(2, 16, 256)
    image_embeds = torch.randn(2, 196, 256)

    decoder = CrossModalFusionDecoder(num_layers=2, hidden_dim=256, num_heads=4)
    fused, attn_map = decoder(text_embeds, image_embeds)

    assert fused.shape == (2, 16, 256)
    assert attn_map is not None
    assert attn_map.shape == (2, 4, 16, 196)


def test_multimodal_vit_forward():
    pixel_values = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 16))
    attn_mask = torch.ones(2, 16)

    model = MultiModalViT(
        hidden_dim=256,
        projection_dim=128,
        num_fusion_layers=2,
        num_heads=4,
        vocab_size=1000,
        vqa_num_classes=50,
        pretrained=False,
    )

    outputs = model(pixel_values, input_ids, attention_mask=attn_mask, task="multitask")

    assert "image_embeds" in outputs
    assert outputs["image_embeds"].shape == (2, 128)
    assert "text_embeds" in outputs
    assert outputs["text_embeds"].shape == (2, 128)
    assert "caption_logits" in outputs
    assert outputs["caption_logits"].shape == (2, 16, 1000)
    assert "vqa_logits" in outputs
    assert outputs["vqa_logits"].shape == (2, 50)
