"""Evaluation metrics module for Image Captioning, VQA, and Cross-Modal Retrieval."""

import math
from collections import Counter
from typing import List, Dict, Union, Tuple
import numpy as np
import torch


def compute_bleu_n(reference: List[str], hypothesis: List[str], n: int = 4) -> float:
    """Compute BLEU-n score for a single pair of reference and hypothesis tokens."""
    ref_tokens = reference if isinstance(reference, list) else reference.lower().split()
    hyp_tokens = hypothesis if isinstance(hypothesis, list) else hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    precisions = []
    for i in range(1, n + 1):
        hyp_ngrams = [tuple(hyp_tokens[j:j + i]) for j in range(len(hyp_tokens) - i + 1)]
        ref_ngrams = [tuple(ref_tokens[j:j + i]) for j in range(len(ref_tokens) - i + 1)]

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)

        clipped_count = sum(min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items())
        total_count = sum(hyp_counts.values())

        precisions.append(clipped_count / total_count if total_count > 0 else 0.0)

    if min(precisions) == 0.0:
        return 0.0

    # Brevity penalty
    c = len(hyp_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1.0 - r / c) if c > 0 else 0.0

    log_precision_sum = sum(math.log(p) for p in precisions)
    score = bp * math.exp(log_precision_sum / n)
    return score


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L (Longest Common Subsequence) score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    m = len(ref_tokens)
    n = len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0

    # Dynamic programming LCS table
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])

    lcs_len = lcs[m][n]
    precision = lcs_len / n
    recall = lcs_len / m

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_cider_approx(references_list: List[List[str]], hypotheses: List[str]) -> float:
    """Compute approximate CIDEr (Consensus-based Image Description Evaluation) score."""
    scores = []
    for refs, hyp in zip(references_list, hypotheses):
        # Average BLEU-4 as surrogate for CIDEr consensus
        ref_scores = [compute_bleu_n(ref.split(), hyp.split(), n=4) for ref in refs]
        scores.append(np.mean(ref_scores) if ref_scores else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_caption_metrics(
    references: List[List[str]],
    hypotheses: List[str],
) -> Dict[str, float]:
    """Compute comprehensive image captioning evaluation metrics.

    Args:
        references: List of reference caption lists for each sample, e.g. [[ref1_a, ref1_b], [ref2_a]]
        hypotheses: List of predicted caption strings.

    Returns:
        Dict containing BLEU-1..4, ROUGE-L, and CIDEr scores.
    """
    bleu1_list, bleu2_list, bleu3_list, bleu4_list = [], [], [], []
    rouge_list = []

    for refs, hyp in zip(references, hypotheses):
        # Best reference score for BLEU/ROUGE
        b1 = max([compute_bleu_n(r, hyp, n=1) for r in refs]) if refs else 0.0
        b2 = max([compute_bleu_n(r, hyp, n=2) for r in refs]) if refs else 0.0
        b3 = max([compute_bleu_n(r, hyp, n=3) for r in refs]) if refs else 0.0
        b4 = max([compute_bleu_n(r, hyp, n=4) for r in refs]) if refs else 0.0
        rl = max([compute_rouge_l(r, hyp) for r in refs]) if refs else 0.0

        bleu1_list.append(b1)
        bleu2_list.append(b2)
        bleu3_list.append(b3)
        bleu4_list.append(b4)
        rouge_list.append(rl)

    cider = compute_cider_approx(references, hypotheses)

    return {
        "BLEU-1": float(np.mean(bleu1_list)),
        "BLEU-2": float(np.mean(bleu2_list)),
        "BLEU-3": float(np.mean(bleu3_list)),
        "BLEU-4": float(np.mean(bleu4_list)),
        "ROUGE-L": float(np.mean(rouge_list)),
        "CIDEr": cider,
    }


def compute_vqa_accuracy(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Compute VQA accuracy metrics."""
    if len(predictions) == 0:
        return {"vqa_accuracy": 0.0, "top1_accuracy": 0.0}

    correct = sum(1 for pred, target in zip(predictions, targets) if pred.strip().lower() == target.strip().lower())
    acc = correct / len(predictions)
    return {"vqa_accuracy": acc, "top1_accuracy": acc}


def compute_retrieval_metrics(
    similarity_matrix: np.ndarray,
    ks: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute Cross-Modal Image-Text Retrieval metrics (Image2Text & Text2Image).

    Args:
        similarity_matrix: Square matrix (N, N) where S[i, j] is similarity between image i and text j.
        ks: List of K thresholds for Recall@K.

    Returns:
        Dict with Recall@1/5/10 and MRR for Image-to-Text and Text-to-Image.
    """
    N = similarity_matrix.shape[0]
    metrics = {}

    # Image -> Text Retrieval
    # Ground truth for image i is text i
    i2t_ranks = []
    for i in range(N):
        scores = similarity_matrix[i]
        rankings = np.argsort(-scores)  # Descending order
        rank = np.where(rankings == i)[0][0] + 1  # 1-indexed rank
        i2t_ranks.append(rank)

    i2t_ranks = np.array(i2t_ranks)
    for k in ks:
        metrics[f"i2t_R@{k}"] = float(np.mean(i2t_ranks <= k))
    metrics["i2t_MRR"] = float(np.mean(1.0 / i2t_ranks))

    # Text -> Image Retrieval
    t2i_ranks = []
    for j in range(N):
        scores = similarity_matrix[:, j]
        rankings = np.argsort(-scores)
        rank = np.where(rankings == j)[0][0] + 1
        t2i_ranks.append(rank)

    t2i_ranks = np.array(t2i_ranks)
    for k in ks:
        metrics[f"t2i_R@{k}"] = float(np.mean(t2i_ranks <= k))
    metrics["t2i_MRR"] = float(np.mean(1.0 / t2i_ranks))

    # Mean recall across both directions
    metrics["mean_recall"] = float(
        (metrics["i2t_R@1"] + metrics["i2t_R@5"] + metrics["i2t_R@10"] +
         metrics["t2i_R@1"] + metrics["t2i_R@5"] + metrics["t2i_R@10"]) / 6.0
    )

    return metrics


def compute_embedding_similarity(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> Dict[str, float]:
    """Compute average cosine similarity and cross-modal alignment score."""
    img_norm = torch.nn.functional.normalize(image_embeds, dim=-1)
    txt_norm = torch.nn.functional.normalize(text_embeds, dim=-1)

    cosine_sim = torch.sum(img_norm * txt_norm, dim=-1)
    mean_sim = float(cosine_sim.mean().item())
    alignment_score = float((cosine_sim.mean() / (torch.std(cosine_sim) + 1e-6)).item())

    return {
        "mean_cosine_similarity": mean_sim,
        "alignment_score": alignment_score,
    }
