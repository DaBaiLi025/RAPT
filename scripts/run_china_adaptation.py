from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _common import write_json
from rapt.config import ModelConfig, TrainConfig
from rapt.data import (
    ArrayStandardizer, PreparedData, TargetStandardizer, load_npz,
    prepare_with_train_only_scalers, temporal_masks,
)
from rapt.metrics import regression_metrics
from rapt.model import MultiSourceRAPT
from rapt.train import choose_device, set_seed, train_one


def load_blob(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def source_scaled_data(data, blob, target_scaler):
    scalers = {
        key: ArrayStandardizer(
            np.asarray(value["mean"], dtype=np.float32),
            np.asarray(value["std"], dtype=np.float32),
        )
        for key, value in blob["feature_scalers"].items()
    }
    return PreparedData(
        veg=scalers["veg"].transform(data.veg),
        prod=scalers["prod"].transform(data.prod),
        climate=scalers["climate"].transform(data.climate),
        aux=scalers["aux"].transform(data.aux),
        y_scaled=target_scaler.transform(data.y),
        y_raw=data.y,
        task_idx=data.task_idx,
        unit_idx=data.unit_idx,
        year=data.year,
        sample_id=data.sample_id,
        y_scaler=target_scaler,
        feature_scalers=scalers,
    )


class TransferredModel(nn.Module):
    def __init__(self, source, config, target_head=False):
        super().__init__()
        self.source = source
        self.target_head_enabled = target_head
        if target_head:
            self.target_head = nn.Sequential(
                nn.Linear(config.fusion_hidden, config.head_hidden), nn.ReLU(),
                nn.Dropout(config.dropout), nn.Linear(config.head_hidden, 1),
            )

    def representation(self, vegetation, productivity, climate, auxiliary):
        z_veg, _ = self.source.veg_enc(vegetation)
        z_prod, _ = self.source.prod_enc(productivity)
        z_cli, _ = self.source.cli_enc(climate)
        z_aux = self.source.aux_enc(auxiliary)
        z = torch.cat([z_veg, z_prod, z_cli, z_aux], dim=1)
        if self.source.use_unit_embedding:
            zeros = torch.zeros(
                z.shape[0], self.source.cfg.unit_embedding_dim,
                device=z.device, dtype=z.dtype,
            )
            z = torch.cat([z, zeros], dim=1)
        return self.source.fusion(z)

    def forward(self, vegetation, productivity, climate, auxiliary):
        shared = self.representation(vegetation, productivity, climate, auxiliary)
        if self.target_head_enabled:
            return self.target_head(shared).squeeze(1)
        predictions = torch.cat([head(shared) for head in self.source.heads], dim=1)
        return predictions.mean(dim=1)


def make_loader(prepared, indices, batch_size, shuffle=False):
    idx = np.asarray(indices, dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(prepared.veg[idx]), torch.from_numpy(prepared.prod[idx]),
        torch.from_numpy(prepared.climate[idx]), torch.from_numpy(prepared.aux[idx]),
        torch.from_numpy(prepared.y_scaled[idx]), torch.from_numpy(idx),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(model, loader, device, target_scaler):
    model.eval()
    targets, predictions, indices = [], [], []
    with torch.no_grad():
        for vegetation, productivity, climate, auxiliary, target, index in loader:
            output = model(
                vegetation.to(device), productivity.to(device), climate.to(device),
                auxiliary.to(device),
            )
            targets.append(target.numpy())
            predictions.append(output.cpu().numpy())
            indices.append(index.numpy())
    return (
        target_scaler.inverse(np.concatenate(targets)),
        target_scaler.inverse(np.concatenate(predictions)),
        np.concatenate(indices),
    )


def save_split_metrics(name, data, predictions, output_dir):
    rows = []
    for split, years in [
        ("train", range(2013, 2021)), ("validation", [2021]),
        ("test", [2022, 2023, 2024]),
    ]:
        mask = np.isin(data.year, list(years))
        rows.append({"strategy": name, "split": split,
                     **regression_metrics(data.y[mask], predictions[mask])})
    pd.DataFrame(rows).to_csv(output_dir / f"{name}_metrics.csv", index=False)
    pd.DataFrame({
        "sample_id": data.sample_id, "year": data.year,
        "observed": data.y, "predicted": predictions,
    }).to_csv(output_dir / f"{name}_predictions.csv", index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-region deployment of RAPT.")
    parser.add_argument("--china-data", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1947)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    data = load_npz(args.china_data)
    train_mask, val_mask, test_mask = temporal_masks(
        data.year, train_end_year=2020, val_year=2021, test_years=(2022, 2023, 2024)
    )
    blob = load_blob(args.source_checkpoint, device)
    config = ModelConfig.from_dict(blob["model_config"])
    source = MultiSourceRAPT(
        veg_dim=data.veg.shape[-1], prod_dim=data.prod.shape[-1],
        cli_dim=data.climate.shape[-1], aux_dim=data.aux.shape[-1],
        n_tasks=len(blob["task_labels"]), n_units=len(blob["unit_labels"]), cfg=config,
    ).to(device)
    source.load_state_dict(blob["model_state"], strict=True)
    for parameter in source.parameters():
        parameter.requires_grad = False

    source_target_scaler = TargetStandardizer(
        float(blob["target_scaler"]["mean"]), float(blob["target_scaler"]["std"])
    )
    direct_data = source_scaled_data(data, blob, source_target_scaler)
    all_indices = np.arange(data.n)
    direct_model = TransferredModel(source, config, target_head=False).to(device)
    _, direct_predictions, _ = predict(
        direct_model, make_loader(direct_data, all_indices, 256), device,
        source_target_scaler,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = save_split_metrics("direct_transfer", data, direct_predictions, args.output_dir)
    slope, intercept = np.polyfit(direct_predictions[train_mask], data.y[train_mask], 1)
    calibrated = slope * direct_predictions + intercept
    all_metrics += save_split_metrics("linear_calibration", data, calibrated, args.output_dir)

    china_target_scaler = TargetStandardizer.fit(data.y[train_mask])
    fine_tune_data = source_scaled_data(data, blob, china_target_scaler)
    fine_tune_model = TransferredModel(source, config, target_head=True).to(device)
    optimizer = torch.optim.AdamW(fine_tune_model.target_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    train_loader = make_loader(fine_tune_data, np.flatnonzero(train_mask), 128, True)
    val_loader = make_loader(fine_tune_data, np.flatnonzero(val_mask), 256)
    best, best_state, wait = np.inf, None, 0
    for _ in range(args.max_epochs):
        fine_tune_model.train()
        for vegetation, productivity, climate, auxiliary, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = fine_tune_model(
                vegetation.to(device), productivity.to(device), climate.to(device),
                auxiliary.to(device),
            )
            loss = criterion(output, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fine_tune_model.target_head.parameters(), 2.0)
            optimizer.step()
        y, p, _ = predict(fine_tune_model, val_loader, device, china_target_scaler)
        rmse = regression_metrics(y, p)["RMSE_kg_ha"]
        if rmse < best:
            best, best_state, wait = rmse, copy.deepcopy(fine_tune_model.state_dict()), 0
        else:
            wait += 1
        if wait >= args.patience:
            break
    fine_tune_model.load_state_dict(best_state)
    _, fine_tuned_predictions, _ = predict(
        fine_tune_model, make_loader(fine_tune_data, all_indices, 256), device,
        china_target_scaler,
    )
    all_metrics += save_split_metrics(
        "target_head_fine_tuning", data, fine_tuned_predictions, args.output_dir
    )
    torch.save({"model_state": best_state}, args.output_dir / "target_head_model.pt")

    china_only_dir = args.output_dir / "china_only_training"
    train_one(
        args.china_data, config, TrainConfig(device=args.device), args.seed,
        china_only_dir, train_end_year=2020, val_year=2021,
        test_years=(2022, 2023, 2024), save_model=True,
    )
    china_only_frame = pd.concat([
        pd.read_csv(china_only_dir / "val_predictions.csv"),
        pd.read_csv(china_only_dir / "test_predictions.csv"),
    ], ignore_index=True)
    for split, years in [("validation", [2021]), ("test", [2022, 2023, 2024])]:
        subset = china_only_frame[china_only_frame.year.isin(years)]
        all_metrics.append({"strategy": "china_only_training", "split": split,
                            **regression_metrics(subset.observed, subset.predicted)})
    pd.DataFrame(all_metrics).to_csv(args.output_dir / "adaptation_metrics.csv", index=False)
    write_json(args.output_dir / "linear_calibration.json", {
        "slope": float(slope), "intercept": float(intercept),
        "fitted_years": "2013-2020",
    })


if __name__ == "__main__":
    main()
