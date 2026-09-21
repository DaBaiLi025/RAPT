from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def per_task_losses(pred, target, task_idx, kind="mse", huber_delta=1.0):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    task_idx = task_idx.reshape(-1).long()

    losses, tasks = [], []
    for t in torch.unique(task_idx):
        m = task_idx == t
        if kind == "mse":
            l = F.mse_loss(pred[m], target[m], reduction="mean")
        elif kind == "huber":
            l = F.huber_loss(
                pred[m], target[m], reduction="mean", delta=float(huber_delta)
            )
        else:
            raise ValueError(kind)
        losses.append(l)
        tasks.append(t)

    return torch.stack(losses), torch.stack(tasks)


class TaskBalancedLoss(nn.Module):
    def __init__(self, base="mse", huber_delta=1.0):
        super().__init__()
        self.base = base
        self.huber_delta = float(huber_delta)

    def forward(self, pred, target, task_idx):
        losses, _ = per_task_losses(
            pred, target, task_idx, self.base, self.huber_delta
        )
        return losses.mean()


class UncertaintyWeightedTaskLoss(nn.Module):
    """
    Learnable homoscedastic uncertainty weighting:
        mean_s [ 0.5 * exp(-q_s) * L_s + 0.5 * q_s ]
    """
    def __init__(self, n_tasks: int, base="mse", huber_delta=1.0):
        super().__init__()
        self.base = base
        self.huber_delta = float(huber_delta)
        self.log_vars = nn.Parameter(torch.zeros(int(n_tasks)))

    def forward(self, pred, target, task_idx):
        losses, tasks = per_task_losses(
            pred, target, task_idx, self.base, self.huber_delta
        )
        q = self.log_vars[tasks.long()]
        return (0.5 * torch.exp(-q) * losses + 0.5 * q).mean()


def build_loss(name: str, n_tasks: int, huber_delta=1.0):
    if name == "tb_mse":
        return TaskBalancedLoss("mse")
    if name == "tb_huber":
        return TaskBalancedLoss("huber", huber_delta)
    if name == "uw_mse":
        return UncertaintyWeightedTaskLoss(n_tasks, "mse")
    if name == "uw_huber":
        return UncertaintyWeightedTaskLoss(n_tasks, "huber", huber_delta)
    raise ValueError(f"Unknown loss: {name}")
