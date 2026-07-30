"""Visual Question Answering (VQA) inference pipeline."""

import os
import sys

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Union, Tuple, Optional
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.multimodal_model import MultiModalViT
from utils.tokenizer import MultiModalTokenizer

# Sample standard VQA candidate answers lookup dictionary
DEFAULT_VQA_ANSWERS = [
    "red", "blue", "green", "yellow", "white", "black", "brown", "pink", "orange", "purple",
    "yes", "no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "dog", "cat", "car", "frisbee", "cushion", "table", "chair", "tree", "grass", "sky", "water",
    "sunny", "cloudy", "rainy", "snowy", "day", "night", "indoor", "outdoor"
]


class VQAEngine:
    """Inference engine for Visual Question Answering."""

    def __init__(
        self,
        model: MultiModalViT,
        tokenizer: Optional[MultiModalTokenizer] = None,
        answer_vocab: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or MultiModalTokenizer()
        self.answer_vocab = answer_vocab or DEFAULT_VQA_ANSWERS
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, Image.Image):
            tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            tensor = image if image.dim() == 4 else image.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image type {type(image)}")
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_answer(
        self,
        image: Union[Image.Image, torch.Tensor],
        question: str,
        top_k: int = 3,
    ) -> Dict[str, Union[str, float, List[Dict[str, Union[str, float]]]]]:
        """Predict top-k answer candidates and confidence scores for a given image and question.

        Args:
            image: PIL Image or Tensor.
            question: Text question string.
            top_k: Number of candidate answers to return.

        Returns:
            Dict containing 'answer', 'confidence', and 'top_k_answers'.
        """
        pixel_values = self.preprocess_image(image)
        encoded_q = self.tokenizer.encode(question)

        input_ids = encoded_q["input_ids"].to(self.device)
        attention_mask = encoded_q["attention_mask"].to(self.device)

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            task="vqa",
        )

        vqa_logits = outputs["vqa_logits"].squeeze(0)
        probs = F.softmax(vqa_logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        top_answers = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            answer_text = self.answer_vocab[idx] if idx < len(self.answer_vocab) else f"candidate_{idx}"
            top_answers.append({"answer": answer_text, "confidence": float(prob)})

        best_answer = top_answers[0]["answer"]
        best_conf = top_answers[0]["confidence"]

        return {
            "answer": best_answer,
            "confidence": best_conf,
            "top_k_answers": top_answers,
        }
