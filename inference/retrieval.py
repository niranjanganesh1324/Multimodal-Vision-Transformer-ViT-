"""Cross-Modal Image-Text Retrieval engine backed by FAISS / PyTorch vector search."""

import os
import sys

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Union, Tuple, Optional
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from models.multimodal_model import MultiModalViT
from utils.tokenizer import MultiModalTokenizer

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class CrossModalRetriever:
    """FAISS and PyTorch vector index for fast sub-millisecond Image-to-Text and Text-to-Image retrieval."""

    def __init__(
        self,
        model: MultiModalViT,
        tokenizer: Optional[MultiModalTokenizer] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or MultiModalTokenizer()
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.image_db: List[Dict[str, Any]] = []
        self.text_db: List[str] = []

        self.image_embeds: Optional[np.ndarray] = None
        self.text_embeds: Optional[np.ndarray] = None

        self.faiss_img_index = None
        self.faiss_txt_index = None

    def preprocess_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, Image.Image):
            tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            tensor = image if image.dim() == 4 else image.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image type {type(image)}")
        return tensor.to(self.device)

    @torch.no_grad()
    def get_image_embedding(self, image: Union[Image.Image, torch.Tensor]) -> np.ndarray:
        pixel_values = self.preprocess_image(image)
        vision_out = self.model.encode_image(pixel_values)
        embed = vision_out["projected_embeds"].cpu().numpy()
        return embed

    @torch.no_grad()
    def get_text_embedding(self, text: str) -> np.ndarray:
        encoded = self.tokenizer.encode(text)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        text_out = self.model.encode_text(input_ids, attention_mask=attention_mask)
        embed = text_out["projected_embeds"].cpu().numpy()
        return embed

    def index_dataset(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        texts: List[str],
        image_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Populate retrieval indices with image and text collections.

        Args:
            images: List of PIL images or Tensors.
            texts: List of text caption strings.
            image_metadata: Optional list of metadata dicts for each image.
        """
        img_embeds_list = [self.get_image_embedding(img) for img in images]
        txt_embeds_list = [self.get_text_embedding(txt) for txt in texts]

        self.image_embeds = np.vstack(img_embeds_list).astype(np.float32)
        self.text_embeds = np.vstack(txt_embeds_list).astype(np.float32)

        self.image_db = image_metadata if image_metadata else [{"id": i} for i in range(len(images))]
        self.text_db = texts

        d_img = self.image_embeds.shape[1]
        d_txt = self.text_embeds.shape[1]

        if HAS_FAISS:
            self.faiss_img_index = faiss.IndexFlatIP(d_img)
            self.faiss_img_index.add(self.image_embeds)

            self.faiss_txt_index = faiss.IndexFlatIP(d_txt)
            self.faiss_txt_index.add(self.text_embeds)
            print(f"[FAISS] Successfully indexed {len(images)} images and {len(texts)} captions.")
        else:
            print(f"[PyTorch Vector Search] Indexed {len(images)} images and {len(texts)} captions (FAISS optional fallback).")

    def search_images_by_text(self, text_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-K matching images for a given text query.

        Args:
            text_query: Search string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with 'metadata', 'score', and 'rank'.
        """
        query_embed = self.get_text_embedding(text_query).astype(np.float32)

        if HAS_FAISS and self.faiss_img_index is not None:
            scores, indices = self.faiss_img_index.search(query_embed, min(top_k, len(self.image_db)))
            scores = scores[0]
            indices = indices[0]
        else:
            # PyTorch Cosine Similarity matrix fallback
            sims = np.dot(self.image_embeds, query_embed.T).squeeze(-1)
            indices = np.argsort(-sims)[:top_k]
            scores = sims[indices]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < len(self.image_db):
                results.append({
                    "rank": rank + 1,
                    "metadata": self.image_db[idx],
                    "score": float(score),
                })
        return results

    def search_texts_by_image(
        self,
        image_query: Union[Image.Image, torch.Tensor],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K matching text captions for a given image query.

        Args:
            image_query: PIL Image or Tensor.
            top_k: Number of results to return.

        Returns:
            List of result dicts with 'text', 'score', and 'rank'.
        """
        query_embed = self.get_image_embedding(image_query).astype(np.float32)

        if HAS_FAISS and self.faiss_txt_index is not None:
            scores, indices = self.faiss_txt_index.search(query_embed, min(top_k, len(self.text_db)))
            scores = scores[0]
            indices = indices[0]
        else:
            sims = np.dot(self.text_embeds, query_embed.T).squeeze(-1)
            indices = np.argsort(-sims)[:top_k]
            scores = sims[indices]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < len(self.text_db):
                results.append({
                    "rank": rank + 1,
                    "text": self.text_db[idx],
                    "score": float(score),
                })
        return results
