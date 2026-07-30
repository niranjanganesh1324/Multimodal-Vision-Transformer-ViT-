"""ONNX & TorchScript Exporter for Multi-Modal Vision Transformer."""

import os
import sys
import argparse

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch

from models.multimodal_model import MultiModalViT


class VisionEncoderWrapper(torch.nn.Module):
    """Wrapper for exporting Vision Encoder contrastive projection to ONNX."""

    def __init__(self, model: MultiModalViT):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model.encode_image(pixel_values)
        return out["projected_embeds"]


class TextEncoderWrapper(torch.nn.Module):
    """Wrapper for exporting Text Encoder contrastive projection to ONNX."""

    def __init__(self, model: MultiModalViT):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model.encode_text(input_ids, attention_mask=attention_mask)
        return out["projected_embeds"]


def export_models(output_dir: str = "./outputs/exported_models"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Export] Initializing MultiModalViT model for export...")

    model = MultiModalViT(hidden_dim=256, projection_dim=128, num_fusion_layers=2, num_heads=4)
    model.eval()

    # Dummy Inputs
    dummy_pixel_values = torch.randn(1, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 1000, (1, 16))
    dummy_attn_mask = torch.ones(1, 16)

    # 1. Export Vision Encoder to ONNX
    vision_wrapper = VisionEncoderWrapper(model)
    onnx_vision_path = os.path.join(output_dir, "vision_encoder.onnx")
    torch.onnx.export(
        vision_wrapper,
        dummy_pixel_values,
        onnx_vision_path,
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "image_embeddings": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"--> Exported Vision Encoder ONNX to '{onnx_vision_path}'")

    # 2. Export Text Encoder to ONNX
    text_wrapper = TextEncoderWrapper(model)
    onnx_text_path = os.path.join(output_dir, "text_encoder.onnx")
    torch.onnx.export(
        text_wrapper,
        (dummy_input_ids, dummy_attn_mask),
        onnx_text_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "text_embeddings": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"--> Exported Text Encoder ONNX to '{onnx_text_path}'")

    # 3. TorchScript Tracing
    traced_vision = torch.jit.trace(vision_wrapper, dummy_pixel_values)
    torchscript_path = os.path.join(output_dir, "vision_encoder_torchscript.pt")
    traced_vision.save(torchscript_path)
    print(f"--> Exported Vision Encoder TorchScript to '{torchscript_path}'")

    print("[Export] All model exports completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MultiModalViT to ONNX & TorchScript")
    parser.add_argument("--output_dir", type=str, default="./outputs/exported_models")
    args = parser.parse_args()
    export_models(args.output_dir)
