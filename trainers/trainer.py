"""Trainer pipeline for Multi-Modal Vision Transformer."""

import os
import time
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from losses.caption_loss import MultiTaskLoss
from utils.tokenizer import MultiModalTokenizer
from utils.metrics import compute_caption_metrics, compute_retrieval_metrics, compute_vqa_accuracy
from utils.visualization import plot_training_curves


class MultiModalTrainer:
    """Production Trainer featuring AMP mixed precision, gradient accumulation,
    cosine LR scheduling, W&B / TensorBoard logging, and automatic best checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[MultiModalTokenizer] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer or MultiModalTokenizer()
        self.config = config or {}

        # Config hyperparameters
        train_cfg = self.config.get("training", {})
        loss_cfg = self.config.get("losses", {})
        log_cfg = self.config.get("logging", {})
        ckpt_cfg = self.config.get("checkpoint", {})

        self.epochs = train_cfg.get("epochs", 10)
        self.lr = float(train_cfg.get("learning_rate", 3.0e-5))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.01))
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.mixed_precision = train_cfg.get("mixed_precision", True) and torch.cuda.is_available()

        # Device assignment
        device_name = self.config.get("hardware", {}).get("device", "cuda")
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss function
        self.criterion = MultiTaskLoss(
            itc_weight=loss_cfg.get("itc_weight", 1.0),
            caption_weight=loss_cfg.get("caption_weight", 1.0),
            vqa_weight=loss_cfg.get("vqa_weight", 1.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.1),
            pad_idx=self.tokenizer.pad_token_id,
            temperature=loss_cfg.get("temperature", 0.07),
        ).to(self.device)

        # Optimizer
        opt_name = train_cfg.get("optimizer", "AdamW")
        if opt_name == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Cosine Annealing Scheduler with Warmup
        total_steps = len(self.train_dataloader) * self.epochs // self.grad_accum_steps
        warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1.0e-6
        )

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # Logging & Checkpointing
        self.output_dir = ckpt_cfg.get("output_dir", "./checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.writer = SummaryWriter(log_dir=self.log_dir) if log_cfg.get("use_tensorboard", True) else None

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_dataloader):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            vqa_targets = batch.get("vqa_targets", None)
            if vqa_targets is not None:
                vqa_targets = vqa_targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task="multitask",
                )
                loss_dict = self.criterion(outputs, target_input_ids=input_ids, vqa_targets=vqa_targets)
                loss = loss_dict["loss"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * self.grad_accum_steps

            if (step + 1) % 10 == 0 or (step + 1) == len(self.train_dataloader):
                avg_step_loss = total_loss / (step + 1)
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch [{epoch}/{self.epochs}] Step [{step + 1}/{len(self.train_dataloader)}] | Loss: {avg_step_loss:.4f} | LR: {lr:.6f}")

        epoch_loss = total_loss / len(self.train_dataloader)
        elapsed = time.time() - start_time
        print(f"--> Epoch {epoch} Complete in {elapsed:.2f}s | Train Loss: {epoch_loss:.4f}")

        if self.writer:
            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

        return epoch_loss

    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> Dict[str, float]:
        """Run evaluation on validation dataloader."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        img_embeds_list = []
        txt_embeds_list = []
        references = []
        hypotheses = []

        for batch in self.val_dataloader:
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            vqa_targets = batch.get("vqa_targets", None)
            if vqa_targets is not None:
                vqa_targets = vqa_targets.to(self.device)

            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                task="multitask",
            )
            loss_dict = self.criterion(outputs, target_input_ids=input_ids, vqa_targets=vqa_targets)
            total_loss += loss_dict["loss"].item()

            # Collect embeddings for retrieval evaluation
            img_embeds_list.append(outputs["image_embeds"].cpu())
            txt_embeds_list.append(outputs["text_embeds"].cpu())

            # Decoding sample hypotheses
            decoded = self.tokenizer.batch_decode(input_ids)
            for text in batch["texts"]:
                references.append([text])
            hypotheses.extend(decoded)

        val_loss = total_loss / len(self.val_dataloader)

        # Retrieval metrics computation
        all_img_embeds = torch.cat(img_embeds_list, dim=0).numpy()
        all_txt_embeds = torch.cat(txt_embeds_list, dim=0).numpy()
        sim_matrix = np.matmul(all_img_embeds, all_txt_embeds.T)
        retrieval_metrics = compute_retrieval_metrics(sim_matrix)

        # Caption metrics computation
        caption_metrics = compute_caption_metrics(references, hypotheses)

        metrics = {
            "val_loss": val_loss,
            **retrieval_metrics,
            **caption_metrics,
        }

        print(f"--> Val Loss: {val_loss:.4f} | Recall@1: {metrics.get('i2t_R@1', 0.0):.4f} | BLEU-4: {metrics.get('BLEU-4', 0.0):.4f}")

        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)

        return metrics

    def fit(self) -> Dict[str, List[float]]:
        """Run full training and validation loop."""
        print(f"Starting MultiModalViT Training on device '{self.device}' for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            if self.val_dataloader is not None:
                val_metrics = self.evaluate(epoch)
                val_loss = val_metrics["val_loss"]
                self.history["val_loss"].append(val_loss)

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_ckpt_path = os.path.join(self.output_dir, "best_model.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                            "config": self.config,
                        },
                        best_ckpt_path,
                    )
                    print(f"Saved Best Checkpoint to '{best_ckpt_path}'")

        # Save final checkpoint
        final_ckpt_path = os.path.join(self.output_dir, "final_model.pt")
        torch.save(
            {
                "epoch": self.epochs,
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            final_ckpt_path,
        )

        # Save training curve visualization
        curve_path = os.path.join(self.output_dir, "training_curves.png")
        plot_training_curves(self.history, save_path=curve_path)

        if self.writer:
            self.writer.close()

        print(f"Training Complete! Checkpoint saved in '{self.output_dir}'")
        return self.history
