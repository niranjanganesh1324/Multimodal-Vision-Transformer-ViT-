"""Unit tests for loss functions."""

import pytest
import torch
from losses.contrastive_loss import InfoNCELoss, TripletMarginLoss
from losses.caption_loss import CaptionLoss, VQALoss, MultiTaskLoss


def test_infonce_loss():
    img = torch.randn(4, 128)
    txt = torch.randn(4, 128)
    img = torch.nn.functional.normalize(img, dim=-1)
    txt = torch.nn.functional.normalize(txt, dim=-1)

    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(img, txt)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_triplet_margin_loss():
    img = torch.randn(4, 128)
    txt = torch.randn(4, 128)
    img = torch.nn.functional.normalize(img, dim=-1)
    txt = torch.nn.functional.normalize(txt, dim=-1)

    loss_fn = TripletMarginLoss(margin=0.2)
    loss = loss_fn(img, txt)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_caption_loss():
    logits = torch.randn(2, 10, 1000)
    targets = torch.randint(0, 1000, (2, 10))

    loss_fn = CaptionLoss(pad_idx=0, label_smoothing=0.1)
    loss = loss_fn(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0


def test_multitask_loss():
    outputs = {
        "image_embeds": torch.nn.functional.normalize(torch.randn(2, 128), dim=-1),
        "text_embeds": torch.nn.functional.normalize(torch.randn(2, 128), dim=-1),
        "caption_logits": torch.randn(2, 10, 1000),
        "vqa_logits": torch.randn(2, 20),
    }
    target_ids = torch.randint(0, 1000, (2, 10))
    vqa_targets = torch.tensor([1, 5])

    loss_fn = MultiTaskLoss()
    res = loss_fn(outputs, target_input_ids=target_ids, vqa_targets=vqa_targets)

    assert "loss" in res
    assert "loss_itc" in res
    assert "loss_caption" in res
    assert "loss_vqa" in res
    assert res["loss"].item() > 0.0
