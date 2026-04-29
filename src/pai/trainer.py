"""Training loop with checkpointing, early stopping, and W&B logging."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
import wandb

from .config import TrainConfig
from .data import build_dataloader
from .losses import build_criterion
from .metrics import get_mask, masked_mae, masked_psnr, masked_ssim
from .models import build_model
from .utils import get_logger, resolve_device, set_seed


def _autogen_run_name(cfg: TrainConfig) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_label = cfg.arch if cfg.arch == "attention_unet" else cfg.backbone
    aug = "aug" if cfg.augment else "noaug"
    return f"{arch_label}_{cfg.loss_type}_{aug}_{ts}"


def _save_checkpoint(path: Path, payload: Dict) -> None:
    torch.save(payload, path)


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)

        self.run_name = cfg.run_name or _autogen_run_name(cfg)
        self.save_dir = Path(cfg.save_root) / self.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Archive config
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        self.logger = get_logger("pai.trainer", self.save_dir / "training_log.txt")
        self.logger.info("Run: %s", self.run_name)
        self.logger.info("Device: %s", self.device)
        self.logger.info("Config: %s", asdict(cfg))

        self.train_loader = build_dataloader("train", cfg)
        self.val_loader = build_dataloader("val", cfg)
        self.logger.info(
            "Data: train=%d  val=%d  (batch_size=%d)",
            len(self.train_loader.dataset),
            len(self.val_loader.dataset),
            cfg.batch_size,
        )

        self.model = build_model(cfg.arch, n_class=cfg.n_class, backbone=cfg.backbone).to(self.device)
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Model: %s | trainable params: %d", cfg.arch, total)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.criterion = build_criterion(cfg, self.device)

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for p, a, _ in self.train_loader:
            p, a = p.to(self.device), a.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(p)
            loss = self.criterion(pred, a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()
            total += loss.item() * p.size(0)
            n += p.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        sums = {"mae": 0.0, "psnr": 0.0, "ssim": 0.0}
        n = 0
        viz = None
        for p, a, _ in self.val_loader:
            p, a = p.to(self.device), a.to(self.device)
            pred = self.model(p)
            mask = get_mask(a)
            sums["mae"] += masked_mae(pred, a, mask).item()
            sums["psnr"] += masked_psnr(pred, a, mask).item()
            sums["ssim"] += masked_ssim(pred, a, mask).item()
            n += 1
            if viz is None:
                viz = (p.detach().cpu(), a.detach().cpu(), pred.detach().cpu())
        avg = {k: v / max(n, 1) for k, v in sums.items()}
        avg["_viz"] = viz
        return avg

    def _log_viz(self, viz) -> "wandb.Image | None":
        if viz is None:
            return None

        def to_disp(t, idx, normalize):
            img = t[0, idx].numpy()
            if normalize:
                return (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img

        p, gt, pr = viz
        row755 = np.concatenate([to_disp(p, 0, True), to_disp(gt, 0, False), to_disp(pr, 0, False)], axis=1)
        row808 = np.concatenate([to_disp(p, 1, True), to_disp(gt, 1, False), to_disp(pr, 1, False)], axis=1)
        grid = np.concatenate([row755, row808], axis=0)
        return wandb.Image(grid, caption="Top: 755nm (In, GT, Pred) | Bottom: 808nm (In, GT, Pred)")

    def fit(self) -> None:
        cfg = self.cfg
        wandb.init(
            project=cfg.wandb_project,
            name=self.run_name,
            config=asdict(cfg),
            mode=cfg.wandb_mode,
        )

        best = {"mae": float("inf"), "psnr": 0.0, "ssim": 0.0}
        best_epoch = {"mae": 0, "psnr": 0, "ssim": 0}
        no_improve = 0

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            val = self._validate()
            self.scheduler.step(val["mae"])

            log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": val["mae"],
                "val_psnr": val["psnr"],
                "val_ssim": val["ssim"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            viz_img = self._log_viz(val["_viz"])
            if viz_img is not None:
                log["Validation_Grid"] = viz_img
            wandb.log(log)

            dt = time.time() - t0
            self.logger.info(
                "Epoch %d/%d | train=%.5f  val_mae=%.5f  psnr=%.2f  ssim=%.4f  (%.1fs)",
                epoch, cfg.epochs, train_loss, val["mae"], val["psnr"], val["ssim"], dt,
            )

            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_mae": val["mae"],
                "val_psnr": val["psnr"],
                "val_ssim": val["ssim"],
                "config": asdict(cfg),
            }
            _save_checkpoint(self.save_dir / "checkpoint_latest.pth", ckpt)

            improved = False
            if val["mae"] < best["mae"]:
                best["mae"] = val["mae"]; best_epoch["mae"] = epoch; improved = True
                _save_checkpoint(self.save_dir / "checkpoint_best_mae.pth", ckpt)
                self.logger.info("  New best MAE: %.6f", best["mae"])
            if val["psnr"] > best["psnr"]:
                best["psnr"] = val["psnr"]; best_epoch["psnr"] = epoch
                _save_checkpoint(self.save_dir / "checkpoint_best_psnr.pth", ckpt)
                self.logger.info("  New best PSNR: %.2f", best["psnr"])
            if val["ssim"] > best["ssim"]:
                best["ssim"] = val["ssim"]; best_epoch["ssim"] = epoch; improved = True
                _save_checkpoint(self.save_dir / "checkpoint_best_ssim.pth", ckpt)
                self.logger.info("  New best SSIM: %.6f", best["ssim"])

            no_improve = 0 if improved else no_improve + 1
            if no_improve >= cfg.early_stop_patience:
                self.logger.info("Early stopping at epoch %d (no improvement for %d).",
                                 epoch, cfg.early_stop_patience)
                break

        self.logger.info("=" * 50)
        self.logger.info("Training complete.")
        self.logger.info("  Best MAE  : %.6f (epoch %d)", best["mae"], best_epoch["mae"])
        self.logger.info("  Best PSNR : %.2f    (epoch %d)", best["psnr"], best_epoch["psnr"])
        self.logger.info("  Best SSIM : %.6f (epoch %d)", best["ssim"], best_epoch["ssim"])
        wandb.finish()
