
## Files

- `us_model_inputs/us_rapt_model_inputs_safe.npz`: model-ready source arrays in a non-pickle NPZ archive. The `state` and `unit` identifiers were converted from Python object arrays to Unicode strings; all numerical arrays and their order are preserved.
- `us_model_inputs/us_sample_index_and_yield.csv`: one row per county-year with sample ID, state FIPS, county FIPS, year, and observed maize yield in kg/ha. Predictor arrays remain in the NPZ file.

## Dataset scope

- 8,376 county-year observations from 762 counties in 12 U.S. states, 2013–2023.
- Temporal split used in the formal U.S. evaluation: 6,094 training observations (2013–2020), 762 validation observations (2021), and 1,520 test observations (2022–2023).
- Dynamic inputs are POS-centered 21-step sequences at relative days −80 to +80 in 8-day increments.
- The archived arrays contain the extracted predictor values before model-time imputation and standardization. Missing predictor values are represented as `NaN`; the training code fits imputation and scaling statistics on the training split only.

## Array dictionary

| Key | Shape | Contents |
|---|---:|---|
| `veg` | `[8376, 21, 8]` | NDVI, EVI2, LSWI, NIRv, GCVI, WDRVI, LAI, FPAR |
| `prod` | `[8376, 21, 2]` | GPP, PsnNet |
| `climate` | `[8376, 21, 9]` | Tmean, Tmax, Tmin, VPD, SoilWater, PRE, RAD, GDD, HeatDay |
| `aux` | `[8376, 6]` | yield_lag1, yield_lag2, yield_lag3, yield_mean_prev, yield_std_prev, year_centered |
| `yield` | `[8376]` | Observed maize yield, kg/ha |
| `state` | `[8376]` | State FIPS code, stored as text |
| `unit` | `[8376]` | County FIPS code, stored as text to preserve leading zeros |
| `year` | `[8376]` | Observation year |
| `sample_id` | `[8376]` | County-year identifier |
| `rel_days` | `[21]` | POS-relative time coordinate in days |
| `*_feature_names` | variable length | Feature order corresponding to each predictor array |

Feature units, product versions, spatial aggregation, and processing provenance must be documented against the final manuscript Methods and the cited source products before public deposition. The public source products themselves are not redistributed here.


