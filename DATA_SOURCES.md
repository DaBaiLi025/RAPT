# Data sources

No yield observations, remote-sensing inputs, meteorological inputs, processed
sample files or model checkpoints are included in this repository.

## Yield statistics

U.S. county-level maize yield records can be downloaded from the United States
Department of Agriculture National Agricultural Statistics Service Quick Stats
database:

https://quickstats.nass.usda.gov/

Northeast China maize yield records can be obtained from statistical yearbooks
and tables published by the provincial statistical bureaus:

- Heilongjiang Provincial Bureau of Statistics: https://tjj.hlj.gov.cn/
- Jilin Provincial Bureau of Statistics: https://tjj.jl.gov.cn/
- Liaoning Provincial Bureau of Statistics: https://tjj.ln.gov.cn/

The provincial websites do not provide a single harmonized interface. Tables
may require manual retrieval, unit conversion, administrative-name matching and
cross-year consistency checks before use.

## Predictor data

The model inputs were derived from MODIS MCD43A4 surface reflectance, MODIS
MCD15A3H LAI/FPAR, MODIS MOD17A2HGF productivity and ERA5-Land meteorological
data. U.S. maize masks were based on the USDA Cropland Data Layer. China maize
masks were based on the published 30 m maize-distribution products cited in the
associated manuscript. These products must be obtained from their respective
providers and used under the applicable licenses.

## Expected processed format

The scripts operate on phenology-aligned, administrative-unit-level samples.
Users are responsible for reproducing the spatial aggregation, peak-of-season
alignment and data harmonization described in the Methods. The required NPZ
schema and the long-table converter are documented in `README.md` and
`scripts/prepare_us_input.py`.
