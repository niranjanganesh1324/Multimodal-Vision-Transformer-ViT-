"""Contrastive loss functions for Image-Text feature alignment."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE / Image-Text Contrastive (ITC) Loss with learnable temperature."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        temp_parameter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            image_embeds: L2-normalized image embeddings tensor (B, D).
            text_embeds: L2-normalized text embeddings tensor (B, D).
            temp_parameter: Optional learnable temperature tensor scalar.

        Returns:
            Scalar contrastive loss tensor.
        """
        tau = temp_parameter if temp_parameter is not None else self.temperature
        tau = torch.clamp(tau, min=0.01, max=1.0)

        # Cosine similarity matrix: (B, B)
        sim_matrix = torch.matmul(image_embeds, text_embeds.T) / tau
        batch_size = image_embeds.shape[0]

        labels = torch.arange(batch_size, device=image_embeds.device)

        # Image-to-Text loss + Text-to-Image loss
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.T, labels)

        return (loss_i2t + loss_t2i) / 2.0


class TripletMarginLoss(nn.Module):
    """Triplet Margin Loss with hard negative mining for image-text representations."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Compute Triplet Margin loss with max hard negative mining."""
        scores = torch.matmul(image_embeds, text_embeds.T)
        diagonal = scores.diag().view(-1, 1)

        # Hardest negative text for each image
        cost_s = self.margin + scores - diagonal
        cost_s = torch.clamp(cost_s, min=0.0)

        # Hardest negative image for each text
        cost_g = self.margin + scores - diagonal.T
        cost_g = torch.clamp(cost_g, min=0.0)

        # Zero out diagonal self-matches
        mask = torch.eye(scores.size(0), device=scores.device).bool()
        cost_s = cost_s.masked_fill(mask, 0.0)
        cost_g = cost_g.masked_fill(mask, 0.0)

        loss = cost_s.max(dim=1)[0].mean() + cost_g.max(dim=0)[0].mean()
        return loss
