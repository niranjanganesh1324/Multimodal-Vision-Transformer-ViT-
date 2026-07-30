"""Captioning, VQA, and Multi-Task Loss functions."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.contrastive_loss import InfoNCELoss


class CaptionLoss(nn.Module):
    """Cross-Entropy captioning loss with label smoothing and pad token masking."""

    def __init__(self, pad_idx: int = 0, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Compute autoregressive caption loss.

        Args:
            logits: Unnormalized token prediction logits tensor (B, S, Vocab).
            target_ids: Ground truth target token IDs tensor (B, S).

        Returns:
            Scalar caption loss tensor.
        """
        # Shift logits and targets for next-token prediction
        # Predict target_ids[:, 1:] using logits[:, :-1]
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_targets = target_ids[:, 1:].contiguous().view(-1)

        loss = self.criterion(shift_logits, shift_targets)
        return loss


class VQALoss(nn.Module):
    """Visual Question Answering Loss over multiple-choice / candidate answer vocabulary."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """Compute VQA classification loss.

        Args:
            logits: Output logits of shape (B, Num_Classes).
            target_labels: Target class indices of shape (B,) or soft targets (B, Num_Classes).

        Returns:
            Scalar VQA loss.
        """
        if target_labels.dim() > 1:
            # Soft targets / multi-answer BCE or KL Div
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(target_labels * log_probs).sum(dim=-1).mean()
        else:
            loss = self.criterion(logits, target_labels)
        return loss


class MultiTaskLoss(nn.Module):
    """Combined Multi-Task Loss incorporating Image-Text Contrastive, Captioning, and VQA losses."""

    def __init__(
        self,
        itc_weight: float = 1.0,
        caption_weight: float = 1.0,
        vqa_weight: float = 1.0,
        label_smoothing: float = 0.1,
        pad_idx: int = 0,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.itc_weight = itc_weight
        self.caption_weight = caption_weight
        self.vqa_weight = vqa_weight

        self.itc_loss = InfoNCELoss(temperature=temperature)
        self.caption_loss = CaptionLoss(pad_idx=pad_idx, label_smoothing=label_smoothing)
        self.vqa_loss = VQALoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_input_ids: torch.Tensor,
        vqa_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total weighted multi-task loss.

        Args:
            outputs: Dictionary returned by MultiModalViT.forward().
            target_input_ids: Ground truth text token IDs tensor (B, S).
            vqa_targets: Optional VQA answer label tensor (B,).

        Returns:
            Dict containing total_loss and individual loss breakdown components.
        """
        losses = {}
        total = torch.tensor(0.0, device=target_input_ids.device)

        # 1. Contrastive Loss (ITC)
        if "image_embeds" in outputs and "text_embeds" in outputs:
            l_itc = self.itc_loss(
                outputs["image_embeds"],
                outputs["text_embeds"],
                temp_parameter=outputs.get("temperature", None),
            )
            losses["loss_itc"] = l_itc
            total = total + self.itc_weight * l_itc

        # 2. Captioning Loss
        if "caption_logits" in outputs:
            l_cap = self.caption_loss(outputs["caption_logits"], target_input_ids)
            losses["loss_caption"] = l_cap
            total = total + self.caption_weight * l_cap

        # 3. VQA Loss
        if "vqa_logits" in outputs and vqa_targets is not None:
            l_vqa = self.vqa_loss(outputs["vqa_logits"], vqa_targets)
            losses["loss_vqa"] = l_vqa
            total = total + self.vqa_weight * l_vqa

        losses["loss"] = total
        return losses
