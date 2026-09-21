from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from _common import paired_unit_bootstrap
from rapt.config import ModelConfig, TrainConfig
from rapt.train import train_one


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPT regional-component ablation.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1947)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    full = ModelConfig()
    variants = {
        "Full": full,
        "Shared_head": replace(full, shared_head=True),
        "Without_unit_embedding": replace(full, use_unit_embedding=False),
        "Without_regional_architecture": replace(
            full, shared_head=True, use_unit_embedding=False
        ),
    }
    train_config = TrainConfig(
        device=args.device, max_epochs=args.max_epochs, patience=args.patience
    )
    rows = []
    for name, config in variants.items():
        print(f"Training {name}", flush=True)
        result = train_one(
            args.data,
            config,
            train_config,
            args.seed,
            args.output_dir / name,
            train_end_year=2020,
            val_year=2021,
            test_years=(2022, 2023),
            save_model=True,
        )
        rows.append({"configuration": name, **result})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)

    full_csv = args.output_dir / "Full" / "test_predictions.csv"
    bootstrap = []
    for name in list(variants)[1:]:
        result = paired_unit_bootstrap(
            full_csv,
            args.output_dir / name / "test_predictions.csv",
            args.bootstrap_iterations,
        )
        bootstrap.append({"configuration": name, **result})
    pd.DataFrame(bootstrap).to_csv(args.output_dir / "bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
