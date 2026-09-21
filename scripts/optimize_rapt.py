from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna
import pandas as pd

from _common import ROOT
from rapt.config import ModelConfig, TrainConfig
from rapt.train import train_one


LOSS_CHOICES = ["tb_mse", "tb_huber", "uw_mse", "uw_huber"]


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def suggest_cfg(trial, use_unit_embedding: bool):
    cfg = ModelConfig(
        veg_hidden=trial.suggest_categorical(
            "veg_hidden", [64, 96, 128, 160]
        ),
        prod_hidden=trial.suggest_categorical(
            "prod_hidden", [32, 48, 64, 96]
        ),
        cli_hidden=trial.suggest_categorical("cli_hidden", [64, 96, 128, 160]),
        num_layers=trial.suggest_int("num_layers", 1, 2),
        aux_hidden=trial.suggest_categorical("aux_hidden", [16, 32, 64]),
        fusion_hidden=trial.suggest_categorical("fusion_hidden", [128, 192, 256, 384]),
        head_hidden=trial.suggest_categorical("head_hidden", [16, 32, 64, 96]),
        dropout=trial.suggest_float("dropout", 0.05, 0.35),

        use_unit_embedding=use_unit_embedding,
        unit_embedding_dim=32,

        use_modality_gating=trial.suggest_categorical(
            "use_modality_gating", [False, True]
        ),
        gating_dim=trial.suggest_categorical("gating_dim", [64, 96, 128]),

        loss_name=trial.suggest_categorical("loss_name", LOSS_CHOICES),
        huber_delta=1.0,

        balanced_task_sampler=True,
        samples_per_task=trial.suggest_categorical(
            "samples_per_task", [4, 6, 8, 10, 12]
        ),

        optimizer_name="adamw",
        lr=trial.suggest_float("lr", 3e-5, 7e-4, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        grad_clip_norm=trial.suggest_categorical(
            "grad_clip_norm", [1.0, 2.0, 5.0]
        ),
        scheduler_factor=trial.suggest_categorical(
            "scheduler_factor", [0.3, 0.5, 0.7]
        ),
    )
    if "huber" in cfg.loss_name:
        cfg.huber_delta = trial.suggest_categorical(
            "huber_delta", [0.5, 1.0, 1.5, 2.0]
        )
    return cfg


def frozen_trial_to_cfg(t, use_unit_embedding: bool):
    d = dict(t.params)
    d["use_unit_embedding"] = use_unit_embedding
    d["unit_embedding_dim"] = 32
    d["balanced_task_sampler"] = True
    d["optimizer_name"] = "adamw"
    if "huber_delta" not in d:
        d["huber_delta"] = 1.0
    return ModelConfig.from_dict(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=8)

    ap.add_argument("--screen-seeds", nargs="+", type=int, default=[3101, 3102])
    ap.add_argument("--rerank-seeds", nargs="+", type=int, default=[4101, 4102, 4103])
    ap.add_argument("--final-seeds", nargs="+", type=int, default=[1947])

    ap.add_argument("--rolling-val-years", nargs="+", type=int,
                    default=[2019, 2020, 2021])

    ap.add_argument("--max-epochs", type=int, default=350)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--scheduler-patience", type=int, default=12)
    ap.add_argument("--device", type=str, default="auto")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--use-unit-embedding", action="store_true")
    g.add_argument("--no-unit-embedding", action="store_true")

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_unit_embedding = not args.no_unit_embedding
    if args.use_unit_embedding:
        use_unit_embedding = True

    tcfg = TrainConfig(
        max_epochs=args.max_epochs,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        device=args.device,
    )

    storage = f"sqlite:///{(args.output_dir / 'optuna.db').as_posix()}"
    study = optuna.create_study(
        study_name="RAPT",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=20260909, multivariate=True, group=True
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=15, n_warmup_steps=8
        ),
    )

    def objective(trial):
        cfg = suggest_cfg(trial, use_unit_embedding)
        vals = []
        for seed_idx, seed in enumerate(args.screen_seeds):
            rd = (
                args.output_dir / "stage1"
                / f"trial_{trial.number:04d}"
                / f"seed_{seed}"
            )
            res = train_one(
                args.data, cfg, tcfg, seed, rd,
                train_end_year=2020,
                val_year=2021,
                test_years=None,
                skip_test=True,
                save_model=False,
                optuna_trial=trial,
                optuna_step_offset=seed_idx * args.max_epochs,
            )
            vals.append(res["val_RMSE_kg_ha"])

        trial.set_user_attr("seed_val_rmse", vals)
        return float(np.mean(vals))

    study.optimize(objective, n_trials=args.trials)

    study.trials_dataframe().to_csv(
        args.output_dir / "stage1_trials.csv",
        index=False, encoding="utf-8-sig"
    )

    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    complete.sort(key=lambda t: float(t.value))
    top = complete[:args.top_k]
    if not top:
        raise RuntimeError("No completed trials.")

    rerank_rows = []
    candidates = []

    for rank, t in enumerate(top, 1):
        cfg = frozen_trial_to_cfg(t, use_unit_embedding)
        vals = []

        for vy in args.rolling_val_years:
            for seed in args.rerank_seeds:
                rd = (
                    args.output_dir / "stage2"
                    / f"rank_{rank:02d}_trial_{t.number:04d}"
                    / f"val_{vy}" / f"seed_{seed}"
                )
                res = train_one(
                    args.data, cfg, tcfg, seed, rd,
                    train_end_year=int(vy) - 1,
                    val_year=int(vy),
                    test_years=None,
                    skip_test=True,
                    save_model=False,
                    optuna_trial=None,
                )
                vals.append(res["val_RMSE_kg_ha"])

        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        row = {
            "rank": rank,
            "trial": t.number,
            "stage1_val_RMSE": float(t.value),
            "rerank_mean_RMSE": m,
            "rerank_sd_RMSE": sd,
            **cfg.to_dict(),
        }
        rerank_rows.append(row)
        candidates.append((m, sd, t.number, cfg))

    rerank_df = pd.DataFrame(rerank_rows).sort_values(
        ["rerank_mean_RMSE", "rerank_sd_RMSE"]
    )
    rerank_df.to_csv(
        args.output_dir / "stage2_rerank.csv",
        index=False, encoding="utf-8-sig"
    )

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    best_mean, best_sd, best_trial, best_cfg = candidates[0]

    selected = {
        "selected_trial": int(best_trial),
        "rerank_mean_RMSE": best_mean,
        "rerank_sd_RMSE": best_sd,
        "model_config": best_cfg.to_dict(),
        "train_config": tcfg.to_dict(),
        "selection": (
            "Validation only. Stage-1 uses 2021; Stage-2 uses rolling "
            "2019/2020/2021 validation. 2022-2023 test is untouched "
            "until this configuration file is written."
        ),
    }
    save_json(args.output_dir / "selected_configuration.json", selected)

    final_rows = []

    for seed in args.final_seeds:
        rd = args.output_dir / "final_run" / f"seed_{seed}"
        res = train_one(
            args.data, best_cfg, tcfg, seed, rd,
            train_end_year=2020,
            val_year=2021,
            test_years=[2022, 2023],
            skip_test=False,
            save_model=True,
        )
        final_rows.append(res)

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(
        args.output_dir / "final_metrics.csv",
        index=False, encoding="utf-8-sig"
    )

    summary = {}
    for metric in ["test_R2", "test_RMSE_kg_ha", "test_MAE_kg_ha"]:
        vals = final_df[metric].astype(float).to_numpy()
        summary[metric] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "values": vals.tolist(),
        }
    save_json(args.output_dir / "final_summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
