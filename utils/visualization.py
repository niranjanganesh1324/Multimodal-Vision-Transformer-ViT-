"""Visualization module for loss curves, attention heatmaps, and embedding spaces."""

import os
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot training and validation loss & accuracy curves.

    Args:
        history: Dictionary containing keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Optional file path to save the generated figure.
        show: Whether to display the plot interactively.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation Progress", fontsize=14, fontweight="bold")

    # Plot Loss
    ax_loss = axes[0]
    if "train_loss" in history:
        ax_loss.plot(history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history:
        ax_loss.plot(history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    ax_loss.set_title("Loss Trajectory")
    ax_loss.set_xlabel("Epochs / Steps")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # Plot Metrics / Accuracy
    ax_acc = axes[1]
    metric_keys = [k for k in history.keys() if k not in ("train_loss", "val_loss")]
    colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, key in enumerate(metric_keys):
        color = colors[idx % len(colors)]
        ax_acc.plot(history[key], label=key, color=color, linewidth=2)
    ax_acc.set_title("Performance Metrics")
    ax_acc.set_xlabel("Epochs / Steps")
    ax_acc.set_ylabel("Score")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fig


def visualize_attention_heatmap(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    attention_map: np.ndarray,
    title: str = "Cross-Attention Heatmap",
    alpha: float = 0.5,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Overlay cross-attention heatmap on an input image.

    Args:
        image: PIL Image, Numpy Array (H, W, 3), or Tensor.
        attention_map: 2D numpy array of shape (H_patch, W_patch) or reshaped attention weights.
        title: Plot title string.
        alpha: Heatmap blend transparency (0.0 to 1.0).
        save_path: Optional output path.

    Returns:
        Overlaid RGB numpy image array.
    """
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().permute(1, 2, 0).numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.array(image)

    H, W = img_np.shape[:2]

    # Normalize attention map
    att = np.array(attention_map)
    att = att - att.min()
    if att.max() > 0:
        att = att / att.max()

    # Resize attention map to image dimensions using PIL or nearest neighbor
    att_img = Image.fromarray((att * 255).astype(np.uint8)).resize((W, H), resample=Image.BICUBIC)
    att_resized = np.array(att_img) / 255.0

    # Apply JET colormap
    cmap = plt.get_cmap("jet")
    heatmap = cmap(att_resized)[:, :, :3]  # Drop alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend image and heatmap
    blended = (img_np * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)

    if save_path:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(blended)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return blended


def plot_embeddings_tsne(
    image_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Visualize visual and textual joint embeddings using t-SNE reduction.

    Args:
        image_embeds: Numpy array of shape (N, D).
        text_embeds: Numpy array of shape (N, D).
        labels: Optional list of sample labels or text queries.
        save_path: Optional output path.
        show: Whether to display plot interactively.

    Returns:
        Matplotlib Figure object.
    """
    from sklearn.manifold import TSNE

    N = image_embeds.shape[0]
    combined = np.concatenate([image_embeds, text_embeds], axis=0)

    tsne = TSNE(n_components=2, perplexity=min(30, max(5, N - 1)), random_state=42)
    embeds_2d = tsne.fit_transform(combined)

    img_2d = embeds_2d[:N]
    txt_2d = embeds_2d[N:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(img_2d[:, 0], img_2d[:, 1], c="#1f77b4", label="Image Embeddings", alpha=0.7, s=60, marker="o")
    ax.scatter(txt_2d[:, 0], txt_2d[:, 1], c="#ff7f0e", label="Text Embeddings", alpha=0.7, s=60, marker="^")

    # Connect paired image-text points with subtle dashed lines
    for i in range(N):
        ax.plot([img_2d[i, 0], txt_2d[i, 0]], [img_2d[i, 1], txt_2d[i, 1]], "gray", linestyle="--", alpha=0.3)
        if labels and i < len(labels) and i < 10:  # Annotate first 10 for clarity
            ax.annotate(labels[i][:15] + "...", (img_2d[i, 0], img_2d[i, 1]), fontsize=8, alpha=0.8)

    ax.set_title("t-SNE Joint Embedding Space Visualization", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fig
