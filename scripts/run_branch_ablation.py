from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import pandas as pd

from _common import paired_unit_bootstrap, write_json
from rapt import data as data_module
from rapt import train as train_module
from rapt.config import ModelConfig, TrainConfig


VARIANTS = {
    "Full": None,
    "Without_vegetation": "veg",
    "Without_productivity": "prod",
    "Without_climate": "climate",
    "Without_auxiliary": "aux",
}


def prepare_masked(data, train_mask, branch):
    prepared = data_module.prepare_with_train_only_scalers(data, train_mask)
    if branch is not None:
        getattr(prepared, branch)[...] = 0.0
    return prepared


@contextlib.contextmanager
def use_branch_mask(branch):
    original = train_module.prepare_with_train_only_scalers
    train_module.prepare_with_train_only_scalers = (
        lambda data, train_mask: prepare_masked(data, train_mask, branch)
    )
    try:
        yield
    finally:
        train_module.prepare_with_train_only_scalers = original


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPT input-branch ablation.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1947)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    model_config = ModelConfig()
    train_config = TrainConfig(
        device=args.device, max_epochs=args.max_epochs, patience=args.patience
    )
    rows = []
    for name, branch in VARIANTS.items():
        run_dir = args.output_dir / name
        print(f"Training {name}", flush=True)
        with use_branch_mask(branch):
            result = train_module.train_one(
                args.data,
                model_config,
                train_config,
                args.seed,
                run_dir,
                train_end_year=2020,
                val_year=2021,
                test_years=(2022, 2023),
                save_model=True,
            )
        rows.append({"configuration": name, **result})

    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "metrics.csv", index=False)
    bootstrap = []
    full_csv = args.output_dir / "Full" / "test_predictions.csv"
    for name in list(VARIANTS)[1:]:
        result = paired_unit_bootstrap(
            full_csv,
            args.output_dir / name / "test_predictions.csv",
            args.bootstrap_iterations,
        )
        bootstrap.append({"configuration": name, **result})
    pd.DataFrame(bootstrap).to_csv(args.output_dir / "bootstrap.csv", index=False)
    write_json(args.output_dir / "feature_groups.json", {
        "Vegetation": ["NDVI", "EVI2", "LSWI", "NIRv", "GCVI", "WDRVI", "LAI", "FPAR"],
        "Productivity": ["GPP", "PsnNet"],
        "Climate": ["Tmean", "Tmax", "Tmin", "VPD", "SoilWater", "PRE", "RAD", "GDD", "HeatDay"],
        "Auxiliary": ["Ylag1", "Ylag2", "Ylag3", "Ymean", "Ystd", "Tyear"],
    })


if __name__ == "__main__":
    main()
