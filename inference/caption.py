"""Image Captioning generator supporting Beam Search, Top-k, Top-p, and Greedy decoding."""

from typing import List, Dict, Union, Optional
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.multimodal_model import MultiModalViT
from utils.tokenizer import MultiModalTokenizer


class ImageCaptioner:
    """Inference pipeline for auto-regressive image captioning."""

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

    def preprocess_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Preprocess PIL image or tensor into batch format (1, 3, 224, 224)."""
        if isinstance(image, Image.Image):
            tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            tensor = image if image.dim() == 4 else image.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image type {type(image)}")
        return tensor.to(self.device)

    @torch.no_grad()
    def generate_caption(
        self,
        image: Union[Image.Image, torch.Tensor],
        decoding_strategy: str = "beam_search",  # Options: beam_search, greedy, sample
        beam_size: int = 5,
        max_length: int = 30,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_captions: int = 1,
    ) -> Union[str, List[str]]:
        """Generate captions for an input image.

        Args:
            image: PIL Image or Tensor.
            decoding_strategy: 'beam_search', 'greedy', or 'sample'.
            beam_size: Beam width for beam search.
            max_length: Maximum caption sequence length.
            temperature: Sampling temperature.
            top_k: Top-k filter cutoff.
            top_p: Top-p (nucleus) threshold.
            num_captions: Number of candidate captions to return.

        Returns:
            Single string caption or list of generated strings.
        """
        pixel_values = self.preprocess_image(image)
        vision_out = self.model.encode_image(pixel_values)
        image_embeds = vision_out["patch_tokens"]

        if decoding_strategy == "greedy":
            return self._greedy_decode(image_embeds, max_length)
        elif decoding_strategy == "sample":
            return self._sample_decode(image_embeds, max_length, temperature, top_k, top_p, num_captions)
        else:
            return self._beam_search_decode(image_embeds, beam_size, max_length, num_captions)

    def _greedy_decode(self, image_embeds: torch.Tensor, max_length: int) -> str:
        """Greedy autoregressive decoding."""
        generated = [self.tokenizer.cls_token_id]

        for _ in range(max_length):
            input_ids = torch.tensor([generated], device=self.device)
            causal_mask = self.model.generate_causal_mask(input_ids.shape[1], device=self.device)
            text_out = self.model.text_encoder(input_ids)

            fused, _ = self.model.fusion_decoder(
                text_embeds=text_out["last_hidden_state"],
                image_embeds=image_embeds,
                text_causal_mask=causal_mask,
            )

            logits = self.model.caption_head(fused[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)

            if next_token == self.tokenizer.sep_token_id:
                break

        return self.tokenizer.decode(generated)

    def _sample_decode(
        self,
        image_embeds: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        num_captions: int,
    ) -> Union[str, List[str]]:
        """Top-k and Top-p Nucleus Sampling decoding."""
        captions = []

        for _ in range(num_captions):
            generated = [self.tokenizer.cls_token_id]

            for _ in range(max_length):
                input_ids = torch.tensor([generated], device=self.device)
                causal_mask = self.model.generate_causal_mask(input_ids.shape[1], device=self.device)
                text_out = self.model.text_encoder(input_ids)

                fused, _ = self.model.fusion_decoder(
                    text_embeds=text_out["last_hidden_state"],
                    image_embeds=image_embeds,
                    text_causal_mask=causal_mask,
                )

                logits = self.model.caption_head(fused[:, -1, :]) / max(temperature, 1.0e-5)

                # Top-K Filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                # Top-P Nucleus Filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

                if next_token == self.tokenizer.sep_token_id:
                    break

            captions.append(self.tokenizer.decode(generated))

        return captions[0] if num_captions == 1 else captions

    def _beam_search_decode(
        self,
        image_embeds: torch.Tensor,
        beam_size: int,
        max_length: int,
        num_captions: int,
    ) -> Union[str, List[str]]:
        """Beam search decoding for optimal caption selection."""
        beams = [([self.tokenizer.cls_token_id], 0.0)]  # (sequence, cumulative_log_prob)

        for _ in range(max_length):
            new_beams = []
            all_completed = True

            for seq, score in beams:
                if seq[-1] == self.tokenizer.sep_token_id:
                    new_beams.append((seq, score))
                    continue

                all_completed = False
                input_ids = torch.tensor([seq], device=self.device)
                causal_mask = self.model.generate_causal_mask(input_ids.shape[1], device=self.device)
                text_out = self.model.text_encoder(input_ids)

                fused, _ = self.model.fusion_decoder(
                    text_embeds=text_out["last_hidden_state"],
                    image_embeds=image_embeds,
                    text_causal_mask=causal_mask,
                )

                logits = self.model.caption_head(fused[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)

                for lp, token in zip(topk_log_probs.tolist(), topk_tokens.tolist()):
                    new_beams.append((seq + [token], score + lp))

            beams = sorted(new_beams, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_size]

            if all_completed:
                break

        results = [self.tokenizer.decode(seq) for seq, _ in beams[:num_captions]]
        return results[0] if num_captions == 1 else results
