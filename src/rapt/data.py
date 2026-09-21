from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


ALIASES = {
    "veg": ["vegetation", "veg", "X_veg"],
    "prod": ["productivity", "prod", "X_prod"],
    "climate": ["climate", "cli", "X_cli", "X_climate"],
    "aux": ["aux", "X_aux", "auxiliary"],
    "yield": ["yield", "y", "target"],
    "state": ["state", "state_id", "task", "task_id"],
    "unit": ["unit", "unit_id", "county", "county_id", "admin_unit"],
    "year": ["year", "years"],
    "sample_id": ["sample_id", "id", "sample"],
}


def _pick(npz, logical_name: str, required: bool = True):
    for k in ALIASES[logical_name]:
        if k in npz.files:
            return npz[k]
    if required:
        raise KeyError(
            f"Missing required array '{logical_name}'. "
            f"Accepted keys: {ALIASES[logical_name]}"
        )
    return None


def _factorize(values):
    arr = np.asarray(values)
    uniques, inverse = np.unique(arr.astype(str), return_inverse=True)
    return inverse.astype(np.int64), uniques.astype(str)


@dataclass
class RAPTData:
    veg: np.ndarray
    prod: np.ndarray
    climate: np.ndarray
    aux: np.ndarray
    y: np.ndarray
    task_idx: np.ndarray
    unit_idx: np.ndarray
    year: np.ndarray
    sample_id: np.ndarray
    task_labels: np.ndarray
    unit_labels: np.ndarray

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_tasks(self) -> int:
        return int(len(self.task_labels))

    @property
    def n_units(self) -> int:
        return int(len(self.unit_labels))


def load_npz(path: Path) -> RAPTData:
    """
    Load RAPT NPZ.

    Predictor NaNs are allowed. They are imputed later using ONLY the active
    training split, which is important for rolling temporal validation.
    Target NaNs/Infs are not allowed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as npz:
        veg = np.asarray(_pick(npz, "veg"), dtype=np.float32)
        prod = np.asarray(_pick(npz, "prod"), dtype=np.float32)
        climate = np.asarray(_pick(npz, "climate"), dtype=np.float32)
        aux = np.asarray(_pick(npz, "aux"), dtype=np.float32)
        y = np.asarray(_pick(npz, "yield"), dtype=np.float32).reshape(-1)
        state_raw = _pick(npz, "state")
        unit_raw = _pick(npz, "unit")
        year = np.asarray(_pick(npz, "year"), dtype=np.int64).reshape(-1)
        sample_id = _pick(npz, "sample_id", required=False)

    n = len(y)
    if sample_id is None:
        sample_id = np.asarray([f"sample_{i:06d}" for i in range(n)], dtype=object)
    else:
        sample_id = np.asarray(sample_id).reshape(-1)

    task_idx, task_labels = _factorize(state_raw)
    unit_idx, unit_labels = _factorize(unit_raw)

    arrays = {
        "veg": veg, "prod": prod, "climate": climate, "aux": aux,
        "y": y, "task_idx": task_idx, "unit_idx": unit_idx,
        "year": year, "sample_id": sample_id,
    }

    for name, a in arrays.items():
        if len(a) != n:
            raise ValueError(f"{name}: first dimension {len(a)} != target length {n}")

    if veg.ndim != 3 or prod.ndim != 3 or climate.ndim != 3:
        raise ValueError("veg/prod/climate must all be 3-D arrays: [N, T, F].")
    if aux.ndim != 2:
        raise ValueError("aux must be 2-D: [N, F_aux].")
    if not (veg.shape[1] == prod.shape[1] == climate.shape[1]):
        raise ValueError("veg/prod/climate must have the same temporal length T.")

    expected_dims = {"vegetation": 8, "productivity": 2, "climate": 9, "aux": 6}
    actual_dims = {
        "vegetation": int(veg.shape[-1]),
        "productivity": int(prod.shape[-1]),
        "climate": int(climate.shape[-1]),
        "aux": int(aux.shape[-1]),
    }
    if actual_dims != expected_dims:
        raise ValueError(
            f"Unexpected input dimensions: {actual_dims}; expected {expected_dims}."
        )
    if int(veg.shape[1]) != 21:
        raise ValueError(f"Expected 21 temporal steps; received {veg.shape[1]}.")

    if not np.isfinite(y).all():
        bad = int((~np.isfinite(y)).sum())
        raise ValueError(f"yield contains {bad} NaN/Inf values.")

    # Infinities in predictors are never meaningful; convert them to NaN so they
    # follow the same train-only imputation path.
    for a in (veg, prod, climate, aux):
        a[np.isinf(a)] = np.nan

    return RAPTData(
        veg=veg, prod=prod, climate=climate, aux=aux, y=y,
        task_idx=task_idx, unit_idx=unit_idx, year=year,
        sample_id=sample_id, task_labels=task_labels, unit_labels=unit_labels
    )


@dataclass
class ArrayStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit_sequence(cls, x: np.ndarray):
        """
        Fit per-feature mean/std over N and T, ignoring NaNs.
        Must be called on training rows only.
        """
        mean = np.nanmean(x, axis=(0, 1), keepdims=True)
        std = np.nanstd(x, axis=(0, 1), keepdims=True)

        if np.isnan(mean).any():
            bad = np.flatnonzero(np.isnan(mean).reshape(-1)).tolist()
            raise ValueError(
                f"Some sequence features are entirely missing in the active "
                f"training split. Feature indices: {bad}"
            )

        std = np.where((std < 1e-8) | np.isnan(std), 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    @classmethod
    def fit_matrix(cls, x: np.ndarray):
        """
        Fit per-feature mean/std over N, ignoring NaNs.
        Must be called on training rows only.
        """
        mean = np.nanmean(x, axis=0, keepdims=True)
        std = np.nanstd(x, axis=0, keepdims=True)

        if np.isnan(mean).any():
            bad = np.flatnonzero(np.isnan(mean).reshape(-1)).tolist()
            raise ValueError(
                f"Some auxiliary features are entirely missing in the active "
                f"training split. Feature indices: {bad}"
            )

        std = np.where((std < 1e-8) | np.isnan(std), 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Train-mean imputation followed by z-score scaling.

        Missing values become zero after standardization because they are first
        replaced with the active training split mean.
        """
        x = np.asarray(x, dtype=np.float32)
        filled = np.where(np.isnan(x), self.mean, x)
        return ((filled - self.mean) / self.std).astype(np.float32)


@dataclass
class TargetStandardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, y: np.ndarray):
        mean = float(np.mean(y))
        std = float(np.std(y))
        if std < 1e-8:
            std = 1.0
        return cls(mean, std)

    def transform(self, y):
        return ((np.asarray(y) - self.mean) / self.std).astype(np.float32)

    def inverse(self, y):
        return np.asarray(y) * self.std + self.mean


@dataclass
class PreparedData:
    veg: np.ndarray
    prod: np.ndarray
    climate: np.ndarray
    aux: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    task_idx: np.ndarray
    unit_idx: np.ndarray
    year: np.ndarray
    sample_id: np.ndarray
    y_scaler: TargetStandardizer
    feature_scalers: Dict[str, ArrayStandardizer]


def prepare_with_train_only_scalers(
    data: RAPTData,
    train_mask: np.ndarray,
) -> PreparedData:
    """
    Fit all feature imputation/scaling statistics using ONLY the active
    training split, then apply unchanged to validation/test rows.
    """
    if not np.any(train_mask):
        raise ValueError("Training split is empty.")

    scalers = {
        "veg": ArrayStandardizer.fit_sequence(data.veg[train_mask]),
        "prod": ArrayStandardizer.fit_sequence(data.prod[train_mask]),
        "climate": ArrayStandardizer.fit_sequence(data.climate[train_mask]),
        "aux": ArrayStandardizer.fit_matrix(data.aux[train_mask]),
    }
    ysc = TargetStandardizer.fit(data.y[train_mask])

    return PreparedData(
        veg=scalers["veg"].transform(data.veg),
        prod=scalers["prod"].transform(data.prod),
        climate=scalers["climate"].transform(data.climate),
        aux=scalers["aux"].transform(data.aux),
        y_scaled=ysc.transform(data.y),
        y_raw=data.y.astype(np.float32),
        task_idx=data.task_idx,
        unit_idx=data.unit_idx,
        year=data.year,
        sample_id=data.sample_id,
        y_scaler=ysc,
        feature_scalers=scalers,
    )


def temporal_masks(
    years,
    *,
    train_end_year: int = 2020,
    val_year: int = 2021,
    test_years: Optional[Sequence[int]] = (2022, 2023),
):
    years = np.asarray(years)

    train_mask = years <= int(train_end_year)
    val_mask = years == int(val_year)

    if test_years is None:
        test_mask = np.zeros_like(train_mask, dtype=bool)
    else:
        test_mask = np.isin(years, np.asarray(test_years, dtype=int))

    return train_mask, val_mask, test_mask


class RAPTDataset(Dataset):
    def __init__(self, prepared: PreparedData, indices: np.ndarray):
        self.p = prepared
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return {
            "veg": torch.from_numpy(self.p.veg[j]),
            "prod": torch.from_numpy(self.p.prod[j]),
            "climate": torch.from_numpy(self.p.climate[j]),
            "aux": torch.from_numpy(self.p.aux[j]),
            "y": torch.tensor(self.p.y_scaled[j], dtype=torch.float32),
            "y_raw": torch.tensor(self.p.y_raw[j], dtype=torch.float32),
            "task": torch.tensor(self.p.task_idx[j], dtype=torch.long),
            "unit": torch.tensor(self.p.unit_idx[j], dtype=torch.long),
            "index": torch.tensor(j, dtype=torch.long),
        }
