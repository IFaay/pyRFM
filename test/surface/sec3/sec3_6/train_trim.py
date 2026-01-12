# -*- coding: utf-8 -*-
"""
Created on 2026/1/12

@author: Yifei Sun
"""
from dataclasses import dataclass
from typing import Literal
from math import inf
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import pyrfm


# ============================================================
# KDTree
# ============================================================
class TorchCKDTree:
    def __init__(self, leafsize=16):
        self.leafsize = leafsize
        self.tree = None
        self.device = None
        self.dtype = None

    def fit(self, x):
        self.device = x.device
        self.dtype = x.dtype
        self.tree = cKDTree(x.detach().cpu().numpy())
        return self

    def query(self, x):
        d, _ = self.tree.query(x.detach().cpu().numpy(), k=1)
        return torch.tensor(d, device=self.device, dtype=self.dtype)


# ============================================================
# BiDirectionalLROnPlateau
# ============================================================
class BiDirectionalLROnPlateau(LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            mode: Literal["min", "max"] = "min",
            *,
            factor: float = 0.5,
            patience: int = 500,
            threshold: float = 1e-4,
            cooldown: int = 200,
            min_lr: float = 1e-6,
            up_factor: float = 1.2,
            up_patience: int = 200,
            up_cooldown: int = 100,
            max_lr: float = 5e-3,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.up_factor = up_factor
        self.patience = patience
        self.up_patience = up_patience
        self.cooldown = cooldown
        self.up_cooldown = up_cooldown
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.threshold = threshold

        self.best = inf if mode == "min" else -inf
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.cooldown_down = 0
        self.cooldown_up = 0

    def step(self, metric: float):
        improved = (
            metric < self.best * (1.0 - self.threshold)
            if self.mode == "min"
            else metric > self.best * (1.0 + self.threshold)
        )

        if improved:
            self.best = metric
            self.num_good_epochs += 1
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            self.num_good_epochs = 0

        if self.cooldown_down > 0:
            self.cooldown_down -= 1
            self.num_bad_epochs = 0

        if self.cooldown_up > 0:
            self.cooldown_up -= 1
            self.num_good_epochs = 0

        if self.num_bad_epochs > self.patience and self.cooldown_down == 0:
            for g in self.optimizer.param_groups:
                g["lr"] = max(g["lr"] * self.factor, self.min_lr)
            self.cooldown_down = self.cooldown
            self.num_bad_epochs = 0

        if self.num_good_epochs > self.up_patience and self.cooldown_up == 0:
            for g in self.optimizer.param_groups:
                g["lr"] = min(g["lr"] * self.up_factor, self.max_lr)
            self.cooldown_up = self.up_cooldown
            self.num_good_epochs = 0


# ============================================================
# Network
# ============================================================
class TwoLayerNet(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, dtype=torch.tensor(0.).dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.tensor(0.).dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, dtype=torch.tensor(0.).dtype),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    pth_path: str = "../../data/airfoil_in.pth"
    nn_epochs: int = 2000
    nn_lr: float = 1e-3
    batch_size: int = 256


# ============================================================
# main
# ============================================================
def main():
    cfg = Config()
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.tensor(0.).device

    # ---------------- data ----------------
    x, normal, _ = torch.load(cfg.pth_path, map_location=device)

    mins = x.min(0).values
    maxs = x.max(0).values
    center = 0.5 * (mins + maxs)
    scale = (maxs - mins).max() * 0.5

    x_in, x_out, x_bnd = torch.load("../../data/airfoil_trim_sets.pth", map_location=device)
    x_in = (x_in - center) / scale
    x_out = (x_out - center) / scale
    x_bnd = (x_bnd - center) / scale

    tree = TorchCKDTree().fit(x_bnd)
    d_in = -tree.query(x_in)
    d_out = tree.query(x_out)

    d_in /= d_in.abs().max()
    d_out /= d_out.abs().max()

    x_train = torch.cat([x_in, x_out], 0)
    y_train = torch.cat([d_in, d_out], 0)

    num_samples = x_train.shape[0]
    batch_size = cfg.batch_size

    model = TwoLayerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.nn_lr)

    # criterion = nn.MSELoss()
    class ClampL1Loss(nn.Module):
        """
        Clamp-L1 loss for signed distance regression.

        loss = mean( min(|pred - target|, delta) )
        """

        def __init__(self, delta: float = 0.1):
            super().__init__()
            self.delta = delta

        def forward(self, pred: torch.Tensor, target: torch.Tensor):
            diff = torch.abs(pred - target)
            return torch.mean(torch.clamp(diff, max=self.delta))

    criterion = ClampL1Loss(delta=0.1)
    scheduler = BiDirectionalLROnPlateau(optimizer)

    eps = 1e-6

    # ---------------- training ----------------
    for epoch in range(cfg.nn_epochs):
        indices = torch.randperm(num_samples, device=device)

        epoch_loss = 0.0
        n = 0

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            x_batch = x_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            y_pred = model(x_batch)

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)
            n += x_batch.size(0)

        epoch_loss /= n
        scheduler.step(epoch_loss)

        if epoch % 200 == 0:
            with torch.no_grad():
                pred_all = model(x_train)

                wrong_sign = (
                        ((pred_all > eps) & (y_train < -eps))
                        | ((pred_all < -eps) & (y_train > eps))
                )
                same_sign = ~wrong_sign
                acc = same_sign.float().mean().item() * 100
                lr = optimizer.param_groups[0]["lr"]

            print(
                f"[Epoch {epoch:5d}] "
                f"loss={epoch_loss:.3e} | "
                f"sign acc={acc:6.2f}% | "
                f"lr={lr:.2e}"
            )


if __name__ == "__main__":
    main()
