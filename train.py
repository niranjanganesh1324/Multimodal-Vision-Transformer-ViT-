"""Main Training CLI Script for Multi-Modal Vision Transformer."""

import os
import argparse
import yaml
from torch.utils.data import DataLoader

from models.multimodal_model import MultiModalViT
from datasets.multimodal_dataset import build_dataset, MultiModalDataCollator
from trainers.trainer import MultiModalTrainer
from utils.tokenizer import MultiModalTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Modal Vision Transformer (ViT)")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml", help="Path to training config YAML")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml", help="Path to model config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset choice (coco, flickr30k, vqa, synthetic)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset fallback mode")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML configs
    with open(args.train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.dataset:
        train_cfg["dataset"]["name"] = args.dataset
    if args.epochs:
        train_cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        train_cfg["training"]["batch_size"] = args.batch_size
    if args.synthetic:
        train_cfg["dataset"]["name"] = "synthetic"

    config = {**train_cfg, "model_config": model_cfg}
    dataset_name = train_cfg["dataset"]["name"]

    print("==================================================================")
    print("      🚀 Multi-Modal Vision Transformer (ViT) Training Pipeline   ")
    print(f"      Dataset: {dataset_name} | Epochs: {train_cfg['training']['epochs']} | Batch Size: {train_cfg['training']['batch_size']}")
    print("==================================================================")

    # Initialize Tokenizer
    tokenizer_name = model_cfg["text_encoder"]["model_name"]
    tokenizer = MultiModalTokenizer(model_name=tokenizer_name)

    # Build Datasets and DataLoaders
    train_dataset = build_dataset(
        dataset_name=dataset_name,
        data_dir=train_cfg["dataset"]["data_dir"],
        split="train",
        synthetic_fallback=train_cfg["dataset"]["synthetic_fallback"],
        num_samples_synthetic=train_cfg["dataset"]["num_samples_synthetic"],
    )
    val_dataset = build_dataset(
        dataset_name=dataset_name,
        data_dir=train_cfg["dataset"]["data_dir"],
        split="val",
        synthetic_fallback=train_cfg["dataset"]["synthetic_fallback"],
        num_samples_synthetic=50,
    )

    collator = MultiModalDataCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Build MultiModalViT Model
    model = MultiModalViT(
        vision_model_name=model_cfg["vision_encoder"]["model_name"],
        text_model_name=model_cfg["text_encoder"]["model_name"],
        hidden_dim=model_cfg["cross_attention"]["hidden_dim"],
        projection_dim=model_cfg["projection"]["projection_dim"],
        num_fusion_layers=model_cfg["cross_attention"]["num_layers"],
        num_heads=model_cfg["cross_attention"]["num_heads"],
        vocab_size=len(tokenizer),
        vqa_num_classes=model_cfg["vqa"]["num_classes"],
        dropout=model_cfg["cross_attention"]["dropout"],
        temperature=train_cfg["losses"]["temperature"],
    )

    # Initialize Trainer and start training
    trainer = MultiModalTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
        config=config,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
