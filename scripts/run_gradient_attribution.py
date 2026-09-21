from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import GradientShap

from _common import write_json
from rapt.config import ModelConfig
from rapt.data import load_npz, prepare_with_train_only_scalers, temporal_masks
from rapt.model import MultiSourceRAPT
from rapt.train import choose_device, set_seed


DYNAMIC_NAMES = [
    "NDVI", "EVI2", "LSWI", "NIRv", "GCVI", "WDRVI", "LAI", "FPAR",
    "GPP", "PsnNet", "Tmean", "Tmax", "Tmin", "VPD", "SoilWater", "PRE",
    "RAD", "GDD", "HeatDay",
]
AUX_NAMES = ["Ylag1", "Ylag2", "Ylag3", "Ymean", "Ystd", "Tyear"]
RELATIVE_DAYS = np.arange(-80, 81, 8)


class YieldWrapper(nn.Module):
    def __init__(self, model, target_mean: float, target_std: float):
        super().__init__()
        self.model = model
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def forward(self, vegetation, productivity, climate, auxiliary, task, unit):
        scaled, _ = self.model(
            vegetation, productivity, climate, auxiliary, task, unit
        )
        return scaled * self.target_std + self.target_mean


def load_checkpoint(path: Path, data, device):
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    config = ModelConfig.from_dict(blob["model_config"])
    model = MultiSourceRAPT(
        veg_dim=data.veg.shape[-1],
        prod_dim=data.prod.shape[-1],
        cli_dim=data.climate.shape[-1],
        aux_dim=data.aux.shape[-1],
        n_tasks=data.n_tasks,
        n_units=data.n_units,
        cfg=config,
    ).to(device)
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval()
    return model


def as_tensors(prepared, indices, device):
    return (
        torch.from_numpy(prepared.veg[indices]).to(device),
        torch.from_numpy(prepared.prod[indices]).to(device),
        torch.from_numpy(prepared.climate[indices]).to(device),
        torch.from_numpy(prepared.aux[indices]).to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GradientSHAP attribution for RAPT.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--background-size", type=int, default=200)
    parser.add_argument("--gradient-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    data = load_npz(args.data)
    train_mask, _, test_mask = temporal_masks(data.year)
    prepared = prepare_with_train_only_scalers(data, train_mask)
    model = load_checkpoint(args.checkpoint, data, device)
    wrapped = YieldWrapper(model, prepared.y_scaler.mean, prepared.y_scaler.std)

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    rng = np.random.default_rng(args.seed)
    background_indices = rng.choice(
        train_indices,
        size=min(args.background_size, len(train_indices)),
        replace=False,
    )
    baselines = as_tensors(prepared, background_indices, device)
    explainer = GradientShap(wrapped)
    dynamic_batches, auxiliary_batches = [], []

    for start in range(0, len(test_indices), args.batch_size):
        indices = test_indices[start : start + args.batch_size]
        inputs = as_tensors(prepared, indices, device)
        task = torch.from_numpy(prepared.task_idx[indices]).long().to(device)
        unit = torch.from_numpy(prepared.unit_idx[indices]).long().to(device)
        attrs = explainer.attribute(
            inputs,
            baselines=baselines,
            n_samples=args.gradient_samples,
            stdevs=0.0,
            additional_forward_args=(task, unit),
        )
        dynamic_batches.append(np.concatenate([
            attrs[0].detach().cpu().numpy(),
            attrs[1].detach().cpu().numpy(),
            attrs[2].detach().cpu().numpy(),
        ], axis=2))
        auxiliary_batches.append(attrs[3].detach().cpu().numpy())

    dynamic = np.concatenate(dynamic_batches)
    auxiliary = np.concatenate(auxiliary_batches)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "dynamic_attributions.npz",
        attribution=dynamic,
        feature_names=np.asarray(DYNAMIC_NAMES),
        relative_days=RELATIVE_DAYS,
        sample_id=data.sample_id[test_indices],
    )
    np.savez_compressed(
        args.output_dir / "auxiliary_attributions.npz",
        attribution=auxiliary,
        feature_names=np.asarray(AUX_NAMES),
        sample_id=data.sample_id[test_indices],
    )

    variable_values = np.abs(dynamic).mean(axis=(0, 1))
    auxiliary_values = np.abs(auxiliary).mean(axis=0)
    variable_table = pd.DataFrame({
        "variable": DYNAMIC_NAMES + AUX_NAMES,
        "mean_absolute_attribution": np.concatenate([variable_values, auxiliary_values]),
        "category": (
            ["Vegetation"] * 8 + ["Productivity"] * 2 + ["Climate"] * 9
            + ["Auxiliary"] * 6
        ),
    }).sort_values("mean_absolute_attribution", ascending=False)
    variable_table.to_csv(args.output_dir / "variable_attribution.csv", index=False)

    temporal = pd.DataFrame({
        "relative_day": RELATIVE_DAYS,
        "mean_absolute_attribution": np.abs(dynamic).mean(axis=(0, 2)),
    })
    temporal.to_csv(args.output_dir / "temporal_attribution.csv", index=False)
    heatmap = pd.DataFrame(
        np.abs(dynamic).mean(axis=0).T,
        index=DYNAMIC_NAMES,
        columns=RELATIVE_DAYS,
    )
    heatmap.to_csv(args.output_dir / "variable_by_time_attribution.csv")
    write_json(args.output_dir / "settings.json", {
        "method": "GradientSHAP",
        "background": "random training samples",
        "background_size": int(len(background_indices)),
        "explained_split": "test_2022_2023",
        "explained_samples": int(len(test_indices)),
        "gradient_samples": args.gradient_samples,
        "output_unit": "kg/ha",
    })


if __name__ == "__main__":
    main()
