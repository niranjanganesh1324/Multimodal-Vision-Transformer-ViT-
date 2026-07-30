"""MS COCO Captions Dataset loader with synthetic fallback for rapid testing."""

import os
import json
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def generate_synthetic_image(width: int = 224, height: int = 224) -> Image.Image:
    """Generate a random synthetic RGB image."""
    array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)


SYNTHETIC_COCO_CAPTIONS = [
    "A brown dog catching a red frisbee in a sunny park.",
    "A sleek modern kitchen with stainless steel appliances and white countertops.",
    "A cute cat sleeping peacefully on a fluffy blue velvet cushion.",
    "A group of people standing on a snow-covered mountain ridge.",
    "A red vintage sports car driving down a coastal highway at sunset.",
    "A plate of delicious hot pizza topped with pepperoni and melting cheese.",
    "A tall lighthouse overlooking crashing ocean waves under cloudy skies.",
    "An aerial view of a bustling city center with skyscrapers and traffic.",
]


class COCODataset(Dataset):
    """MS COCO Captions PyTorch Dataset with synthetic fallback mode."""

    def __init__(
        self,
        data_dir: str = "./data/coco",
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
        self.is_synthetic = False
        self.samples: List[Dict[str, Any]] = []

        # Check if local COCO dataset files exist
        anno_file = os.path.join(data_dir, f"annotations/captions_{split}2017.json")
        img_dir = os.path.join(data_dir, f"{split}2017")

        if os.path.exists(anno_file) and os.path.exists(img_dir):
            try:
                with open(anno_file, "r") as f:
                    data = json.load(f)
                img_dict = {img["id"]: img["file_name"] for img in data["images"]}
                for ann in data["annotations"]:
                    img_id = ann["image_id"]
                    if img_id in img_dict:
                        self.samples.append({
                            "image_path": os.path.join(img_dir, img_dict[img_id]),
                            "caption": ann["caption"],
                            "image_id": img_id,
                        })
            except Exception as e:
                print(f"[Warning] Error reading COCO annotations ({e}). Switching to synthetic mode.")
                self.is_synthetic = True
        else:
            if synthetic_fallback:
                self.is_synthetic = True
            else:
                raise FileNotFoundError(f"COCO dataset not found at {data_dir}. Set synthetic_fallback=True to enable synthetic dataset mode.")

        if self.is_synthetic:
            # Generate in-memory synthetic samples
            for i in range(num_samples_synthetic):
                caption = SYNTHETIC_COCO_CAPTIONS[i % len(SYNTHETIC_COCO_CAPTIONS)]
                self.samples.append({
                    "image_path": None,
                    "caption": f"{caption} (sample #{i + 1})",
                    "image_id": i,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        if self.is_synthetic or item["image_path"] is None or not os.path.exists(item["image_path"]):
            image = generate_synthetic_image()
        else:
            try:
                image = Image.open(item["image_path"]).convert("RGB")
            except Exception:
                image = generate_synthetic_image()

        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "caption": item["caption"],
            "image_id": item["image_id"],
            "raw_image": image,
        }
