"""Evaluation CLI Script for Multi-Modal Vision Transformer."""

import os
import sys
import argparse
import yaml

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

from models.multimodal_model import MultiModalViT
from datasets.multimodal_dataset import build_dataset, MultiModalDataCollator
from trainers.trainer import MultiModalTrainer
from utils.tokenizer import MultiModalTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Modal Vision Transformer")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    dataset_name = "synthetic" if args.synthetic else train_cfg["dataset"]["name"]

    print("==================================================================")
    print("      📊 Multi-Modal Vision Transformer Evaluation Suite         ")
    print(f"      Dataset: {dataset_name} | Checkpoint: {args.checkpoint or 'Initialized Weight Baseline'}")
    print("==================================================================")

    tokenizer = MultiModalTokenizer(model_name=model_cfg["text_encoder"]["model_name"])

    val_dataset = build_dataset(
        dataset_name=dataset_name,
        data_dir=train_cfg["dataset"]["data_dir"],
        split="val",
        synthetic_fallback=True,
        num_samples_synthetic=50,
    )
    collator = MultiModalDataCollator(tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collator)

    model = MultiModalViT(
        vision_model_name=model_cfg["vision_encoder"]["model_name"],
        text_model_name=model_cfg["text_encoder"]["model_name"],
        hidden_dim=model_cfg["cross_attention"]["hidden_dim"],
        projection_dim=model_cfg["projection"]["projection_dim"],
        num_fusion_layers=model_cfg["cross_attention"]["num_layers"],
        num_heads=model_cfg["cross_attention"]["num_heads"],
        vocab_size=len(tokenizer),
        vqa_num_classes=model_cfg["vqa"]["num_classes"],
        pretrained=not args.synthetic,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Checkpoint weights from '{args.checkpoint}'")

    trainer = MultiModalTrainer(
        model=model,
        train_dataloader=val_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
        config={**train_cfg, "model_config": model_cfg},
    )

    metrics = trainer.evaluate()
    print("\n--- Final Evaluation Summary ---")
    for k, v in metrics.items():
        print(f"  • {k}: {v:.4f}")


if __name__ == "__main__":
    main()
