from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, TensorDataset

from _common import write_json
from rapt.data import load_npz, prepare_with_train_only_scalers, temporal_masks
from rapt.metrics import regression_metrics
from rapt.train import choose_device, set_seed


CONFIGS = {
    "LSTM": dict(hidden=160, layers=2, aux=16, fusion=256, head=32,
                 dropout=0.0741, lr=4.7270e-4, weight_decay=9.8608e-4,
                 batch_size=128, grad_clip=1.0, scheduler_factor=0.7),
    "GRU": dict(hidden=96, layers=2, aux=32, fusion=96, head=16,
                dropout=0.3287, lr=5.0960e-4, weight_decay=4.7261e-6,
                batch_size=64, grad_clip=2.0, scheduler_factor=0.5),
    "Transformer": dict(d_model=96, heads=8, layers=1, feedforward=192,
                        aux=16, fusion=192, head=96, dropout=0.1074,
                        lr=4.5544e-4, weight_decay=9.7196e-4,
                        batch_size=64, grad_clip=5.0, scheduler_factor=0.5),
}


class NeuralBaseline(nn.Module):
    def __init__(self, name: str, input_dim: int, aux_dim: int, config: dict):
        super().__init__()
        self.name = name
        if name in {"LSTM", "GRU"}:
            cls = nn.LSTM if name == "LSTM" else nn.GRU
            self.input_projection = nn.Identity()
            self.encoder = cls(
                input_dim, config["hidden"], config["layers"], batch_first=True,
                dropout=config["dropout"] if config["layers"] > 1 else 0.0,
            )
            temporal_dim = config["hidden"]
            self.position = None
        else:
            self.input_projection = nn.Linear(input_dim, config["d_model"])
            layer = nn.TransformerEncoderLayer(
                d_model=config["d_model"], nhead=config["heads"],
                dim_feedforward=config["feedforward"], dropout=config["dropout"],
                batch_first=True, activation="relu",
            )
            self.encoder = nn.TransformerEncoder(layer, config["layers"])
            self.position = nn.Parameter(torch.zeros(1, 21, config["d_model"]))
            temporal_dim = config["d_model"]
        self.auxiliary = nn.Sequential(
            nn.Linear(aux_dim, config["aux"]), nn.ReLU(), nn.Dropout(config["dropout"])
        )
        self.regressor = nn.Sequential(
            nn.Linear(temporal_dim + config["aux"], config["fusion"]),
            nn.ReLU(), nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion"], config["head"]), nn.ReLU(),
            nn.Dropout(config["dropout"]), nn.Linear(config["head"], 1),
        )

    def forward(self, sequence, auxiliary):
        sequence = self.input_projection(sequence)
        if self.position is not None:
            encoded = self.encoder(sequence + self.position[:, : sequence.shape[1]])
            temporal = encoded.mean(dim=1)
        else:
            encoded, _ = self.encoder(sequence)
            temporal = encoded[:, -1]
        return self.regressor(torch.cat([temporal, self.auxiliary(auxiliary)], dim=1)).squeeze(1)


def arrays(prepared):
    sequence = np.concatenate([prepared.veg, prepared.prod, prepared.climate], axis=2)
    return sequence.astype(np.float32), prepared.aux.astype(np.float32)


def predict_neural(model, loader, device, target_scaler):
    model.eval()
    observed, predicted, indices = [], [], []
    with torch.no_grad():
        for sequence, auxiliary, target, index in loader:
            output = model(sequence.to(device), auxiliary.to(device)).cpu().numpy()
            observed.append(target.numpy())
            predicted.append(output)
            indices.append(index.numpy())
    observed = target_scaler.inverse(np.concatenate(observed))
    predicted = target_scaler.inverse(np.concatenate(predicted))
    return observed, predicted, np.concatenate(indices)


def neural_run(name, prepared, masks, output_dir, seed, device, max_epochs, patience):
    train_mask, val_mask, test_mask = masks
    sequence, auxiliary = arrays(prepared)
    config = CONFIGS[name]
    model = NeuralBaseline(name, sequence.shape[2], auxiliary.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config["scheduler_factor"], patience=12
    )
    criterion = nn.MSELoss()

    def loader(mask, shuffle=False):
        idx = np.flatnonzero(mask)
        dataset = TensorDataset(
            torch.from_numpy(sequence[idx]), torch.from_numpy(auxiliary[idx]),
            torch.from_numpy(prepared.y_scaled[idx]), torch.from_numpy(idx),
        )
        return DataLoader(dataset, batch_size=config["batch_size"] if shuffle else 512,
                          shuffle=shuffle)

    train_loader, val_loader, test_loader = loader(train_mask, True), loader(val_mask), loader(test_mask)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for sequence_batch, auxiliary_batch, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(sequence_batch.to(device), auxiliary_batch.to(device)), target.to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
        val_y, val_p, _ = predict_neural(model, val_loader, device, prepared.y_scaler)
        val_rmse = regression_metrics(val_y, val_p)["RMSE_kg_ha"]
        scheduler.step(val_rmse)
        if val_rmse < best:
            best, best_state, wait = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    rows, result = [], {"model": name}
    for split, split_loader in [("validation", val_loader), ("test", test_loader)]:
        y, p, idx = predict_neural(model, split_loader, device, prepared.y_scaler)
        metrics = regression_metrics(y, p)
        result.update({f"{split}_{key}": value for key, value in metrics.items()})
        if split == "test":
            rows = pd.DataFrame({"index": idx, "observed": y, "predicted": p})
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "test_predictions.csv", index=False)
    torch.save({"model_state": best_state, "configuration": config}, output_dir / "best_model.pt")
    write_json(output_dir / "metrics.json", result)
    return result


def tabular_run(name, prepared, masks, output_dir, seed):
    train_mask, val_mask, test_mask = masks
    sequence, auxiliary = arrays(prepared)
    x = np.concatenate([sequence.reshape(len(sequence), -1), auxiliary], axis=1)
    if name == "RF":
        model = RandomForestRegressor(
            n_estimators=800, max_depth=24, min_samples_split=4,
            min_samples_leaf=2, max_features="sqrt", bootstrap=False,
            random_state=seed, n_jobs=-1,
        )
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=600, learning_rate=0.03019, max_depth=4,
            min_child_weight=1.9463, subsample=0.762, colsample_bytree=0.6119,
            gamma=0.7789, reg_alpha=1.5762e-5, reg_lambda=1.4004,
            objective="reg:squarederror", random_state=seed, n_jobs=-1,
        )
    model.fit(x[train_mask], prepared.y_raw[train_mask])
    result = {"model": name}
    for split, mask in [("validation", val_mask), ("test", test_mask)]:
        prediction = model.predict(x[mask])
        result.update({f"{split}_{key}": value for key, value in
                       regression_metrics(prepared.y_raw[mask], prediction).items()})
        if split == "test":
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"index": np.flatnonzero(mask), "observed": prepared.y_raw[mask],
                          "predicted": prediction}).to_csv(
                output_dir / "test_predictions.csv", index=False
            )
    write_json(output_dir / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RAPT reference models.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1947)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    data = load_npz(args.data)
    masks = temporal_masks(data.year)
    prepared = prepare_with_train_only_scalers(data, masks[0])
    results = []
    for name in ["RF", "XGBoost"]:
        results.append(tabular_run(name, prepared, masks, args.output_dir / name, args.seed))
    for name in ["LSTM", "GRU", "Transformer"]:
        results.append(neural_run(
            name, prepared, masks, args.output_dir / name, args.seed, device,
            args.max_epochs, args.patience,
        ))
    pd.DataFrame(results).to_csv(args.output_dir / "baseline_metrics.csv", index=False)


if __name__ == "__main__":
    main()
