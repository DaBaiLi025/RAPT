from __future__ import annotations
import copy
import json
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import (
    RAPTData, load_npz, prepare_with_train_only_scalers,
    temporal_masks, RAPTDataset
)
from .sampler import TaskBalancedBatchSampler
from .losses import build_loss
from .model import MultiSourceRAPT
from .metrics import regression_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility preference; users may turn benchmark on externally if desired.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(s: str):
    if s != "auto":
        return torch.device(s)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate(model, loader, device, y_scaler):
    model.eval()
    ys, ps, idxs = [], [], []
    with torch.no_grad():
        for b in loader:
            for k in ["veg", "prod", "climate", "aux", "y", "task", "unit"]:
                b[k] = b[k].to(device)
            pred, _ = model(
                b["veg"], b["prod"], b["climate"], b["aux"],
                b["task"], b["unit"]
            )
            ys.append(b["y"].detach().cpu().numpy())
            ps.append(pred.detach().cpu().numpy())
            idxs.append(b["index"].numpy())

    y_scaled = np.concatenate(ys)
    p_scaled = np.concatenate(ps)
    idx = np.concatenate(idxs)
    y_raw = y_scaler.inverse(y_scaled)
    p_raw = y_scaler.inverse(p_scaled)
    return regression_metrics(y_raw, p_raw), idx, y_raw, p_raw


def train_one(
    data_path: Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    output_dir: Path,
    *,
    train_end_year: int = 2020,
    val_year: int = 2021,
    test_years: Optional[Sequence[int]] = (2022, 2023),
    skip_test: bool = False,
    save_model: bool = True,
    optuna_trial=None,
    optuna_step_offset: int = 0,
):
    set_seed(int(seed))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(data_path)
    train_mask, val_mask, test_mask = temporal_masks(
        data.year,
        train_end_year=train_end_year,
        val_year=val_year,
        test_years=None if skip_test else test_years,
    )

    if not np.any(train_mask):
        raise ValueError("Empty training split.")
    if not np.any(val_mask):
        raise ValueError(f"No validation samples for year {val_year}.")
    if not skip_test and not np.any(test_mask):
        raise ValueError(f"No test samples for years {test_years}.")

    p = prepare_with_train_only_scalers(data, train_mask)

    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)

    ds_train = RAPTDataset(p, train_idx)
    ds_val = RAPTDataset(p, val_idx)

    if model_cfg.balanced_task_sampler:
        # Dataset-local task labels
        local_tasks = p.task_idx[train_idx]
        batch_sampler = TaskBalancedBatchSampler(
            local_tasks,
            samples_per_task=model_cfg.samples_per_task,
            seed=seed,
        )
        train_loader = DataLoader(
            ds_train,
            batch_sampler=batch_sampler,
            num_workers=train_cfg.num_workers,
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=max(1, data.n_tasks * model_cfg.samples_per_task),
            shuffle=True,
            num_workers=train_cfg.num_workers,
        )

    val_loader = DataLoader(
        ds_val, batch_size=512, shuffle=False, num_workers=train_cfg.num_workers
    )

    device = choose_device(train_cfg.device)

    model = MultiSourceRAPT(
        veg_dim=data.veg.shape[-1],
        prod_dim=data.prod.shape[-1],
        cli_dim=data.climate.shape[-1],
        aux_dim=data.aux.shape[-1],
        n_tasks=data.n_tasks,
        n_units=data.n_units,
        cfg=model_cfg,
    ).to(device)

    criterion = build_loss(
        model_cfg.loss_name, data.n_tasks, model_cfg.huber_delta
    ).to(device)

    params = list(model.parameters())
    loss_params = [p_ for p_ in criterion.parameters() if p_.requires_grad]
    params += loss_params

    if model_cfg.optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=model_cfg.lr, weight_decay=model_cfg.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            params, lr=model_cfg.lr, weight_decay=model_cfg.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=model_cfg.scheduler_factor,
        patience=train_cfg.scheduler_patience,
        min_lr=train_cfg.min_lr,
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []

    for epoch in range(train_cfg.max_epochs):
        model.train()
        criterion.train()
        losses = []

        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        for b in train_loader:
            for k in ["veg", "prod", "climate", "aux", "y", "task", "unit"]:
                b[k] = b[k].to(device)

            optimizer.zero_grad(set_to_none=True)

            pred, _ = model(
                b["veg"], b["prod"], b["climate"], b["aux"],
                b["task"], b["unit"]
            )
            loss = criterion(pred, b["y"], b["task"])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                params, max_norm=float(model_cfg.grad_clip_norm)
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _, _, _ = _evaluate(model, val_loader, device, p.y_scaler)
        val_rmse = float(val_metrics["RMSE_kg_ha"])
        scheduler.step(val_rmse)

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_R2": val_metrics["R2"],
            "val_RMSE_kg_ha": val_metrics["RMSE_kg_ha"],
            "val_MAE_kg_ha": val_metrics["MAE_kg_ha"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if optuna_trial is not None:
            optuna_trial.report(
                val_rmse,
                step=int(optuna_step_offset) + int(epoch),
            )
            if optuna_trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        if val_rmse < best_val - 1e-9:
            best_val = val_rmse
            best_epoch = epoch + 1
            stale = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "criterion": copy.deepcopy(criterion.state_dict()),
            }
        else:
            stale += 1

        if stale >= train_cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("No best state was recorded.")

    model.load_state_dict(best_state["model"])
    criterion.load_state_dict(best_state["criterion"])

    pd.DataFrame(history).to_csv(
        output_dir / "history.csv", index=False, encoding="utf-8-sig"
    )

    val_metrics, val_global_idx, val_y, val_p = _evaluate(
        model, val_loader, device, p.y_scaler
    )

    pd.DataFrame({
        "sample_id": data.sample_id[val_global_idx],
        "year": data.year[val_global_idx],
        "task_idx": data.task_idx[val_global_idx],
        "unit_idx": data.unit_idx[val_global_idx],
        "observed": val_y,
        "predicted": val_p,
    }).to_csv(output_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")

    result = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "val_R2": float(val_metrics["R2"]),
        "val_RMSE_kg_ha": float(val_metrics["RMSE_kg_ha"]),
        "val_MAE_kg_ha": float(val_metrics["MAE_kg_ha"]),
    }

    if not skip_test:
        ds_test = RAPTDataset(p, test_idx)
        test_loader = DataLoader(
            ds_test, batch_size=512, shuffle=False, num_workers=train_cfg.num_workers
        )
        test_metrics, test_global_idx, test_y, test_p = _evaluate(
            model, test_loader, device, p.y_scaler
        )

        pd.DataFrame({
            "sample_id": data.sample_id[test_global_idx],
            "year": data.year[test_global_idx],
            "task_idx": data.task_idx[test_global_idx],
            "unit_idx": data.unit_idx[test_global_idx],
            "observed": test_y,
            "predicted": test_p,
        }).to_csv(
            output_dir / "test_predictions.csv",
            index=False,
            encoding="utf-8-sig"
        )

        result.update({
            "test_R2": float(test_metrics["R2"]),
            "test_RMSE_kg_ha": float(test_metrics["RMSE_kg_ha"]),
            "test_MAE_kg_ha": float(test_metrics["MAE_kg_ha"]),
        })

    if save_model:
        torch.save({
            "model_state": model.state_dict(),
            "criterion_state": criterion.state_dict(),
            "model_config": model_cfg.to_dict(),
            "train_config": train_cfg.to_dict(),
            "task_labels": data.task_labels.tolist(),
            "unit_labels": data.unit_labels.tolist(),
            "target_scaler": {"mean": p.y_scaler.mean, "std": p.y_scaler.std},
            "feature_scalers": {
                k: {"mean": v.mean, "std": v.std}
                for k, v in p.feature_scalers.items()
            },
            "split": {
                "train_end_year": int(train_end_year),
                "val_year": int(val_year),
                "test_years": None if test_years is None else list(test_years),
            },
        }, output_dir / "best_model.pt")

    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
