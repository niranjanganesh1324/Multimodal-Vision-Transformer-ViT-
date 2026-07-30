"""Unified dataset factory and collation module for PyTorch DataLoader."""

from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

from datasets.coco import COCODataset
from datasets.flickr30k import Flickr30KDataset
from datasets.vqa import VQADataset
from utils.tokenizer import MultiModalTokenizer


class MultiModalDataCollator:
    """Collates raw images and text into tokenized PyTorch batch tensors."""

    def __init__(self, tokenizer: MultiModalTokenizer, max_length: int = 64) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values_list = [item["pixel_values"] for item in batch]
        pixel_values = torch.stack(pixel_values_list, dim=0)

        texts = [item.get("caption", item.get("question", "")) for item in batch]
        encoded = self.tokenizer.encode(texts)

        collated = {
            "pixel_values": pixel_values,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "texts": texts,
        }

        if "vqa_target" in batch[0]:
            collated["vqa_targets"] = torch.tensor([item["vqa_target"] for item in batch], dtype=torch.long)

        return collated


def build_dataset(
    dataset_name: str = "coco",
    data_dir: str = "./data",
    split: str = "train",
    synthetic_fallback: bool = True,
    num_samples_synthetic: int = 200,
) -> Dataset:
    """Factory function to build target dataset.

    Args:
        dataset_name: Name of dataset ('coco', 'flickr30k', 'vqa', or 'synthetic').
        data_dir: Root directory path.
        split: Split name ('train', 'val', 'test').
        synthetic_fallback: Whether to use synthetic data fallback if real files missing.
        num_samples_synthetic: Number of synthetic samples.

    Returns:
        PyTorch Dataset instance.
    """
    dataset_name = dataset_name.lower()
    if dataset_name in ("coco", "synthetic"):
        return COCODataset(
            data_dir=os.path.join(data_dir, "coco"),
            split=split,
            synthetic_fallback=synthetic_fallback,
            num_samples_synthetic=num_samples_synthetic,
        )
    elif dataset_name == "flickr30k":
        return Flickr30KDataset(
            data_dir=os.path.join(data_dir, "flickr30k"),
            split=split,
            synthetic_fallback=synthetic_fallback,
            num_samples_synthetic=num_samples_synthetic,
        )
    elif dataset_name == "vqa":
        return VQADataset(
            data_dir=os.path.join(data_dir, "vqa"),
            split=split,
            synthetic_fallback=synthetic_fallback,
            num_samples_synthetic=num_samples_synthetic,
        )
    else:
        raise ValueError(f"Unsupported dataset name '{dataset_name}'. Supported: coco, flickr30k, vqa, synthetic")
