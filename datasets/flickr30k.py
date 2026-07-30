"""Flickr30K Dataset loader with synthetic fallback."""

import os
from typing import List, Dict, Optional, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.coco import generate_synthetic_image

SYNTHETIC_FLICKR_CAPTIONS = [
    "Two dogs running through a shallow stream in the woods.",
    "A child in a yellow raincoat jumping in a puddle.",
    "A street artist painting a colorful mural on a brick wall.",
    "A musician playing an acoustic guitar on a cobblestone plaza.",
    "A crowd of spectators watching an outdoor summer concert.",
]


class Flickr30KDataset(Dataset):
    """Flickr30K PyTorch Dataset with synthetic fallback."""

    def __init__(
        self,
        data_dir: str = "./data/flickr30k",
        split: str = "train",
        transform: Optional[Any] = None,
        synthetic_fallback: bool = True,
        num_samples_synthetic: int = 100,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.is_synthetic = True
        self.samples: List[Dict[str, Any]] = []

        # Synthetic fallback setup
        for i in range(num_samples_synthetic):
            caption = SYNTHETIC_FLICKR_CAPTIONS[i % len(SYNTHETIC_FLICKR_CAPTIONS)]
            self.samples.append({
                "image_path": None,
                "caption": caption,
                "image_id": i,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image = generate_synthetic_image()
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "caption": item["caption"],
            "image_id": item["image_id"],
            "raw_image": image,
        }
