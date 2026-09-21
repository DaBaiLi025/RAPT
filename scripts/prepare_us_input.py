"""
Convert a long-format U.S. POS-centered table into the RAPT NPZ input.

The converter:
- validates exactly 21 POS-centered observations per county-year;
- builds Vegetation, Productivity and Climate tensors;
- builds the six-variable historical-yield auxiliary vector;
- computes lagged yields using exact calendar-year lags (t-1, t-2, t-3);
- computes prior-yield mean/std using ONLY years < t;
- never uses current/future yield to construct historical features;
- preserves missing predictor values as NaN.

Missing predictor values are intentionally NOT globally imputed here.
The RAPT data loader performs imputation separately for each
training/validation protocol using TRAINING-SPLIT-ONLY feature means. This
keeps the rolling validation protocol leakage-safe.

Required columns are documented in README.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


VEGETATION_FEATURES = [
    "NDVI", "EVI2", "LSWI", "NIRv", "GCVI", "WDRVI", "LAI", "FPAR"
]

PRODUCTIVITY_FEATURES = ["GPP", "PsnNet"]

CLIMATE_FEATURES = [
    "Tmean", "Tmax", "Tmin", "VPD", "SoilWater", "PRE", "RAD", "GDD", "HeatDay"
]

AUX_FEATURES = [
    "yield_lag1",
    "yield_lag2",
    "yield_lag3",
    "yield_mean_prev",
    "yield_std_prev",
    "year_centered",
]

EXPECTED_STEPS = np.arange(21, dtype=int)
EXPECTED_REL_DAYS = np.arange(-80, 81, 8, dtype=float)


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns in source CSV: " + ", ".join(missing)
        )


def _format_geoid(v) -> str:
    # U.S. county FIPS/GEOID is 5 digits.
    try:
        return f"{int(float(v)):05d}"
    except Exception:
        s = str(v).strip()
        return s.zfill(5) if s.isdigit() else s


def _format_state(v) -> str:
    try:
        return f"{int(float(v)):02d}"
    except Exception:
        s = str(v).strip()
        return s.zfill(2) if s.isdigit() else s


def _build_sample_table(df: pd.DataFrame, year_center: float) -> pd.DataFrame:
    # One row per county-year.
    sample = (
        df.groupby(["GEOID_str", "year_int"], as_index=False)
          .agg(
              state=("STATEFP_str", "first"),
              yield_kg_ha=("yield_kg_ha", "first"),
              n_rows=("step", "size"),
          )
          .sort_values(["GEOID_str", "year_int"])
          .reset_index(drop=True)
    )

    # Validate target/state consistency inside each county-year.
    chk = (
        df.groupby(["GEOID_str", "year_int"])
          .agg(
              y_nunique=("yield_kg_ha", "nunique"),
              state_nunique=("STATEFP_str", "nunique"),
          )
    )
    if (chk["y_nunique"] != 1).any():
        bad = chk.index[chk["y_nunique"] != 1].tolist()[:10]
        raise ValueError(f"Yield is not constant within county-year groups: {bad}")
    if (chk["state_nunique"] != 1).any():
        bad = chk.index[chk["state_nunique"] != 1].tolist()[:10]
        raise ValueError(f"State is not constant within county-year groups: {bad}")

    # Exact calendar-year lag lookup. This is NOT a row shift, so a missing calendar
    # year remains missing instead of silently using an older available observation.
    lookup = {
        (r.GEOID_str, int(r.year_int)): float(r.yield_kg_ha)
        for r in sample.itertuples(index=False)
    }

    for lag in (1, 2, 3):
        sample[f"yield_lag{lag}"] = [
            lookup.get((g, int(y) - lag), np.nan)
            for g, y in zip(sample["GEOID_str"], sample["year_int"])
        ]

    # Prior-year summaries using only records with year < current year.
    mean_prev = np.full(len(sample), np.nan, dtype=np.float64)
    std_prev = np.full(len(sample), np.nan, dtype=np.float64)

    for _, idx in sample.groupby("GEOID_str", sort=False).groups.items():
        idx = list(idx)
        sub = sample.loc[idx].sort_values("year_int")
        prior: List[float] = []

        for row_idx, y in zip(sub.index, sub["yield_kg_ha"].to_numpy(dtype=float)):
            if prior:
                arr = np.asarray(prior, dtype=np.float64)
                mean_prev[row_idx] = float(arr.mean())
                # Population SD (ddof=0); with one prior record this is 0.
                std_prev[row_idx] = float(arr.std(ddof=0))
            prior.append(float(y))

    sample["yield_mean_prev"] = mean_prev
    sample["yield_std_prev"] = std_prev
    sample["year_centered"] = sample["year_int"].astype(float) - float(year_center)

    return sample


def _validate_temporal_groups(df: pd.DataFrame) -> None:
    group_sizes = df.groupby(["GEOID_str", "year_int"]).size()
    bad_size = group_sizes[group_sizes != 21]
    if len(bad_size):
        raise ValueError(
            "Every county-year must contain exactly 21 rows. "
            f"Found {len(bad_size)} invalid groups; examples: "
            f"{bad_size.head(10).to_dict()}"
        )

    bad_step = []
    bad_rel = []

    for key, g in df.groupby(["GEOID_str", "year_int"], sort=False):
        gs = g.sort_values("step")
        steps = gs["step"].to_numpy(dtype=int)
        rel = gs["rel_day"].to_numpy(dtype=float)

        if not np.array_equal(steps, EXPECTED_STEPS):
            bad_step.append(key)
        if not np.allclose(rel, EXPECTED_REL_DAYS, atol=1e-6, rtol=0):
            bad_rel.append(key)

        if len(bad_step) >= 10 or len(bad_rel) >= 10:
            break

    if bad_step:
        raise ValueError(
            f"Invalid step sequence; expected 0..20. Examples: {bad_step[:10]}"
        )
    if bad_rel:
        raise ValueError(
            "Invalid rel_day sequence; expected -80,-72,...,80. "
            f"Examples: {bad_rel[:10]}"
        )


def _tensorize(
    df: pd.DataFrame,
    sample: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    blocks = []

    indexed = {
        key: g.sort_values("step")
        for key, g in df.groupby(["GEOID_str", "year_int"], sort=False)
    }

    for r in sample.itertuples(index=False):
        g = indexed[(r.GEOID_str, int(r.year_int))]
        block = g[feature_cols].to_numpy(dtype=np.float32)
        if block.shape != (21, len(feature_cols)):
            raise RuntimeError(
                f"Unexpected tensor block shape for {(r.GEOID_str, r.year_int)}: "
                f"{block.shape}"
            )
        blocks.append(block)

    return np.stack(blocks, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Long-format source CSV.")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("us_rapt_input.npz"),
        help="Output NPZ path.",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON conversion report. Default: <output>.report.json",
    )
    ap.add_argument(
        "--train-start-year",
        type=int,
        default=2013,
        help="Used only to define the default year-centering constant.",
    )
    ap.add_argument(
        "--train-end-year",
        type=int,
        default=2020,
        help="Used only to define the default year-centering constant.",
    )
    ap.add_argument(
        "--year-center",
        type=float,
        default=None,
        help=(
            "Constant subtracted from year. Default is the midpoint of the fixed "
            "training years, i.e. mean(2013..2020)=2016.5."
        ),
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    year_center = (
        float(args.year_center)
        if args.year_center is not None
        else float(np.mean(np.arange(args.train_start_year, args.train_end_year + 1)))
    )

    required = (
        ["GEOID", "STATEFP", "year_int", "step", "rel_day", "yield_kg_ha"]
        + VEGETATION_FEATURES
        + PRODUCTIVITY_FEATURES
        + CLIMATE_FEATURES
    )

    df = pd.read_csv(args.input)
    _require_columns(df, required)

    # Normalize key fields.
    df["GEOID_str"] = df["GEOID"].map(_format_geoid)
    df["STATEFP_str"] = df["STATEFP"].map(_format_state)
    df["year_int"] = pd.to_numeric(df["year_int"], errors="raise").astype(int)
    df["step"] = pd.to_numeric(df["step"], errors="raise").astype(int)
    df["rel_day"] = pd.to_numeric(df["rel_day"], errors="raise").astype(float)
    df["yield_kg_ha"] = pd.to_numeric(df["yield_kg_ha"], errors="raise").astype(float)

    # Feature columns must be numeric, but NaNs are allowed.
    for c in VEGETATION_FEATURES + PRODUCTIVITY_FEATURES + CLIMATE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["GEOID_str", "year_int", "step"]).reset_index(drop=True)

    _validate_temporal_groups(df)

    sample = _build_sample_table(df, year_center=year_center)

    vegetation = _tensorize(df, sample, VEGETATION_FEATURES)
    productivity = _tensorize(df, sample, PRODUCTIVITY_FEATURES)
    climate = _tensorize(df, sample, CLIMATE_FEATURES)
    aux = sample[AUX_FEATURES].to_numpy(dtype=np.float32)

    y = sample["yield_kg_ha"].to_numpy(dtype=np.float32)
    state = sample["state"].astype(str).to_numpy()
    unit = sample["GEOID_str"].astype(str).to_numpy()
    year = sample["year_int"].to_numpy(dtype=np.int64)
    sample_id = np.asarray(
        [f"{g}_{yr}" for g, yr in zip(unit, year)],
        dtype=str,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "vegetation": vegetation,
        "productivity": productivity,
        "climate": climate,
        "aux": aux,
        "yield": y,
        "state": state,
        "unit": unit,
        "year": year,
        "sample_id": sample_id,
        "rel_days": EXPECTED_REL_DAYS.astype(np.float32),
        "vegetation_feature_names": np.asarray(VEGETATION_FEATURES, dtype=str),
        "productivity_feature_names": np.asarray(PRODUCTIVITY_FEATURES, dtype=str),
        "climate_feature_names": np.asarray(CLIMATE_FEATURES, dtype=str),
        "aux_feature_names": np.asarray(AUX_FEATURES, dtype=str),
    }
    np.savez_compressed(args.output, **payload)

    # Audit report.
    years_unique = sorted(np.unique(year).astype(int).tolist())
    year_counts = {
        str(int(yr)): int(np.sum(year == yr))
        for yr in years_unique
    }

    split_counts = {
        f"train_{args.train_start_year}_{args.train_end_year}": int(
            np.sum((year >= args.train_start_year) & (year <= args.train_end_year))
        ),
        "validation_2021": int(np.sum(year == 2021)),
        "test_2022_2023": int(np.sum(np.isin(year, [2022, 2023]))),
    }

    def nan_count(a: np.ndarray) -> int:
        return int(np.isnan(a).sum())

    report = {
        "input_csv": str(args.input.resolve()),
        "output_npz": str(args.output.resolve()),
        "n_long_rows": int(len(df)),
        "n_samples": int(len(sample)),
        "n_units": int(sample["GEOID_str"].nunique()),
        "n_states": int(sample["state"].nunique()),
        "years": years_unique,
        "year_counts": year_counts,
        "fixed_protocol_counts": split_counts,
        "tensor_shapes": {
            "vegetation": list(vegetation.shape),
            "productivity": list(productivity.shape),
            "climate": list(climate.shape),
            "aux": list(aux.shape),
            "yield": list(y.shape),
        },
        "features": {
            "vegetation": VEGETATION_FEATURES,
            "productivity": PRODUCTIVITY_FEATURES,
            "climate": CLIMATE_FEATURES,
            "auxiliary": AUX_FEATURES,
        },
        "year_center": year_center,
        "missing_values_preserved": {
            "vegetation": nan_count(vegetation),
            "productivity": nan_count(productivity),
            "climate": nan_count(climate),
            "aux": nan_count(aux),
        },
        "aux_missing_by_feature": {
            c: int(sample[c].isna().sum()) for c in AUX_FEATURES
        },
        "note": (
            "Predictor NaNs are preserved. The RAPT loader must "
            "impute them using the active training split only. Historical-yield "
            "features use only years earlier than the current sample."
        ),
    }

    report_path = (
        args.report
        if args.report is not None
        else args.output.with_suffix(args.output.suffix + ".report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
