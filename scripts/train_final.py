from __future__ import annotations

import argparse
from pathlib import Path

from _common import write_json
from rapt.config import ModelConfig, TrainConfig
from rapt.train import train_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate RAPT.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1947)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    model_config = ModelConfig()
    train_config = TrainConfig(
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "configuration.json",
        {
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
            "split": {"train": "2013-2020", "validation": [2021], "test": [2022, 2023]},
        },
    )
    result = train_one(
        args.data,
        model_config,
        train_config,
        args.seed,
        args.output_dir,
        train_end_year=2020,
        val_year=2021,
        test_years=(2022, 2023),
        save_model=True,
    )
    print(result)


if __name__ == "__main__":
    main()
