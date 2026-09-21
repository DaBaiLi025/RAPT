from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from _common import ROOT
from rapt.data import load_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()

    d = load_npz(args.data)
    print("Input structure OK")
    print(f"N samples: {d.n}")
    print(
        f"vegetation:  {d.veg.shape} | "
        f"NaN={int(np.isnan(d.veg).sum())}"
    )
    print(
        f"productivity: {d.prod.shape} | "
        f"NaN={int(np.isnan(d.prod).sum())}"
    )
    print(f"climate: {d.climate.shape} | NaN={int(np.isnan(d.climate).sum())}")
    print(f"aux:     {d.aux.shape} | NaN={int(np.isnan(d.aux).sum())}")
    print(f"years:   {d.year.min()} - {d.year.max()}")
    print(f"tasks:   {d.n_tasks} -> {d.task_labels.tolist()}")
    print(f"units:   {d.n_units}")
    print("year counts:")
    ys, cs = np.unique(d.year, return_counts=True)
    for y, c in zip(ys, cs):
        print(f"  {y}: {c}")
    print(
        "Note: predictor NaNs are allowed here and are imputed using "
        "active-training-split-only means during model training."
    )


if __name__ == "__main__":
    main()
