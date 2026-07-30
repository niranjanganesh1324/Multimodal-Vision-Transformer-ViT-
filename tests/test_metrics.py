"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
from utils.metrics import (
    compute_bleu_n,
    compute_rouge_l,
    compute_caption_metrics,
    compute_vqa_accuracy,
    compute_retrieval_metrics,
)


def test_bleu_score():
    ref = "a dog playing with a red ball in the grass"
    hyp = "a dog playing with a ball in grass"
    score = compute_bleu_n(ref.split(), hyp.split(), n=4)
    assert 0.0 <= score <= 1.0


def test_rouge_l():
    ref = "the cat sat on the mat"
    hyp = "the cat is sitting on the mat"
    score = compute_rouge_l(ref, hyp)
    assert 0.0 <= score <= 1.0


def test_vqa_accuracy():
    preds = ["red", "dog", "yes", "blue"]
    targets = ["red", "cat", "yes", "blue"]
    acc = compute_vqa_accuracy(preds, targets)
    assert acc["vqa_accuracy"] == 0.75


def test_retrieval_metrics():
    # Identity similarity matrix (perfect retrieval)
    sim_matrix = np.eye(5)
    metrics = compute_retrieval_metrics(sim_matrix, ks=[1, 5])

    assert metrics["i2t_R@1"] == 1.0
    assert metrics["t2i_R@1"] == 1.0
    assert metrics["i2t_MRR"] == 1.0
    assert metrics["mean_recall"] == 1.0
