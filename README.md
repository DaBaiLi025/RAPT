# RAPT

RAPT is a region-aware phenology-aligned temporal fusion model for maize yield
prediction. This repository contains source code only. Study data, trained
weights and generated results are not redistributed.

## Model inputs

Each sample represents one administrative unit and year. Dynamic variables are
stored as 21 observations from 80 days before to 80 days after the
sample-specific peak-of-season date, at 8-day intervals.

- Vegetation: NDVI, EVI2, LSWI, NIRv, GCVI, WDRVI, LAI and FPAR
- Productivity: GPP and PsnNet
- Climate: Tmean, Tmax, Tmin, VPD, SoilWater, PRE, RAD, GDD and HeatDay
- Auxiliary: Ylag1, Ylag2, Ylag3, Ymean, Ystd and Tyear

The NPZ input keys are `vegetation`, `productivity`, `climate`, `aux`,
`yield`, `state`, `unit`, `year` and `sample_id`. Array shapes are
`[N, 21, 8]`, `[N, 21, 2]`, `[N, 21, 9]`, `[N, 6]` and `[N]` for the target.
Predictor missing values may be represented by `NaN`; imputation and
standardization statistics are fitted on the active training data only.

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.
Install a PyTorch build compatible with the available CUDA runtime when GPU
training is required.

## Input preparation

Convert a long-format U.S. table containing one row per relative time step:

```bash
python scripts/prepare_us_input.py --input data/us_long.csv --output data/us_input.npz
python scripts/validate_input.py --data data/us_input.npz
```

Required identifier columns are `GEOID`, `STATEFP`, `year_int`, `step`,
`rel_day` and `yield_kg_ha`, together with the 19 dynamic variables listed
above. Each county-year must contain steps 0 to 20 and relative days -80 to 80.
Historical-yield inputs are generated using records strictly earlier than the
prediction year.

China data use the same NPZ schema. The `state` field identifies province and
the `unit` field identifies the reporting administrative unit.

## Experiments

Train the final model with the chronological U.S. split (2013-2020 training,
2021 validation and 2022-2023 testing):

```bash
python scripts/train_final.py --data data/us_input.npz --output-dir results/rapt
```

Run the reference models, input-branch ablation and regional-component
ablation:

```bash
python scripts/train_baselines.py --data data/us_input.npz --output-dir results/baselines
python scripts/run_branch_ablation.py --data data/us_input.npz --output-dir results/branch_ablation
python scripts/run_regional_ablation.py --data data/us_input.npz --output-dir results/regional_ablation
```

Compute GradientSHAP attribution from a trained checkpoint:

```bash
python scripts/run_gradient_attribution.py --data data/us_input.npz --checkpoint results/rapt/best_model.pt --output-dir results/attribution
```

Run the Northeast China deployment and adaptation analysis:

```bash
python scripts/run_china_adaptation.py --china-data data/china_input.npz --source-checkpoint results/rapt/best_model.pt --output-dir results/china
```

The adaptation script uses 2013-2020 for target-domain training or calibration,
2021 for validation and 2022-2024 for testing. Direct transfer and target-head
fine-tuning use source-training feature transformations. China-only training
fits its transformations on target-domain training data.

Hyperparameter optimization is available through:

```bash
python scripts/optimize_rapt.py --data data/us_input.npz --output-dir results/search
```

Run `python <script> --help` for optional arguments. Experiment outputs are
written outside the tracked source tree by default through `.gitignore`.

## Data availability

See `DATA_SOURCES.md` for the public data providers and preparation boundary.

## License

The source code is provided under the MIT License. Data products remain subject
to the terms of their respective providers.
