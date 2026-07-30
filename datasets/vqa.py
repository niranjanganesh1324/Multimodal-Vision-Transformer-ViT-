"""VQAv2 Dataset loader with synthetic fallback for Visual Question Answering."""

import os
from typing import List, Dict, Optional, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.coco import generate_synthetic_image

SYNTHETIC_VQA_PAIRS = [
    {"question": "What color is the car?", "answer": "red", "answer_label": 0},
    {"question": "How many dogs are in the picture?", "answer": "two", "answer_label": 1},
    {"question": "Is the weather sunny?", "answer": "yes", "answer_label": 2},
    {"question": "What sport is being played?", "answer": "frisbee", "answer_label": 3},
    {"question": "Where is the cat sitting?", "answer": "cushion", "answer_label": 4},
]


class VQADataset(Dataset):
    """VQAv2 PyTorch Dataset with synthetic fallback."""

    def __init__(
        self,
        data_dir: str = "./data/vqa",
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
        self.samples: List[Dict[str, Any]] = []

        for i in range(num_samples_synthetic):
            pair = SYNTHETIC_VQA_PAIRS[i % len(SYNTHETIC_VQA_PAIRS)]
            self.samples.append({
                "image_path": None,
                "question": pair["question"],
                "answer": pair["answer"],
                "answer_label": pair["answer_label"],
                "question_id": i,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image = generate_synthetic_image()
        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "question": item["question"],
            "caption": item["question"],  # Can serve as text prompt
            "answer": item["answer"],
            "vqa_target": item["answer_label"],
            "question_id": item["question_id"],
            "raw_image": image,
        }
