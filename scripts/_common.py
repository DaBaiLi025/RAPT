from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def paired_unit_bootstrap(
    full_csv: Path,
    variant_csv: Path,
    iterations: int,
    seed: int = 2026,
) -> dict:
    full = pd.read_csv(full_csv)
    variant = pd.read_csv(variant_csv)
    keys = ["sample_id", "year", "unit_idx", "observed"]
    paired = full[keys + ["predicted"]].rename(
        columns={"predicted": "full_pred"}
    ).merge(
        variant[keys + ["predicted"]].rename(
            columns={"predicted": "variant_pred"}
        ),
        on=keys,
        validate="one_to_one",
    )
    units = paired["unit_idx"].unique()
    unit_col = paired["unit_idx"].to_numpy()
    observed = paired["observed"].to_numpy(float)
    full_pred = paired["full_pred"].to_numpy(float)
    variant_pred = paired["variant_pred"].to_numpy(float)
    point = float(
        np.sqrt(np.mean((variant_pred - observed) ** 2))
        - np.sqrt(np.mean((full_pred - observed) ** 2))
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for i in range(iterations):
        draw = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([np.flatnonzero(unit_col == unit) for unit in draw])
        y = observed[idx]
        deltas[i] = (
            np.sqrt(np.mean((variant_pred[idx] - y) ** 2))
            - np.sqrt(np.mean((full_pred[idx] - y) ** 2))
        )
    return {
        "delta_RMSE_kg_ha": point,
        "ci95_lower": float(np.quantile(deltas, 0.025)),
        "ci95_upper": float(np.quantile(deltas, 0.975)),
        "bootstrap_iterations": int(iterations),
        "n_units": int(len(units)),
    }
