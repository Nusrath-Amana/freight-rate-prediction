# Freight Rate Prediction Challenge

Predicts `posted_rate` for truckload freight. See `freight-rate-ml-assessment.pdf`
for the assessment instructions and `Freight_Rate_Prediction_Report.pdf` for the
full write-up.

## Install and run

```bash
python -m pip install -r requirements.txt

python -m src.benchmark     # compare backends x model shapes x encodings
python -m src.train         # tune, evaluate, write reports/best_params.json
python -m src.predict       # fit with those params, write both prediction files

python score.py --predictions validation_predictions.csv \
    --december-predictions data/december-chart-inputs.csv
```

Every command runs from the repository root. `src.train` accepts
`--learner {sklearn,xgboost,catboost,lightgbm}`, `--shape {boosted,hybrid}`,
`--trials N` and `--no-tune`.

## Approach

**Target is `posted_rate / distance`** — dollars per mile, not raw dollars. The
date effect fits one number per day and adds it to every load on that day, so
that number has to mean the same thing for a \$600 haul and a \$5,000 one. In
dollars it cannot: the same +1.28% market drift is \$7.60 on the shortest
distance decile and \$63.26 on the longest. In \$/mile the same drift spans only
1.4×. Because distance is a known input, relative error in the rate equals
relative error in the \$/mile, so nothing is lost by predicting the ratio.

**677 corrupted labels are gated out of training.** A row's \$/mile divided by
the median for its peer group (distance ventile × equipment) is bimodal with
genuinely empty gaps at [0.50, 0.75) and [1.25, 2.00): 98.6% of rows sit in
0.87–1.14 and 677 rows (1.41%) sit at 0.17–0.45× or 2–5.3×, spread evenly
across all ten months. That is injected corruption, not a heavy tail. These
rows are dropped from *fits only* and never from anything scored.

**A parametric date effect carries the extrapolation.** The graded set is
November–December; training ends 31 October. A boosted tree handed a day index
reproduces the last bin it saw and flattens out, leaving a forward holdout
biased low. The hybrid model backfits a date-free booster against an OLS daily
effect on `[elapsed days, market_index, day of week]`, which extrapolates. The
drift term is only identified when elapsed time and `market_index` are not
collinear — they are strongly collinear early in the year (+0.88 for Jan–May),
so `trend_collinearity` is measured and the code falls back to the plain
booster above 0.55 rather than assuming the precondition holds. The full
Jan–Oct window sits at −0.109, so the drift is well identified for the shipped
fit.

**Features are constrained by what December ships.**
`december-chart-inputs.csv` carries only `pickup, delivery, distance,
equipment, weight, date` — no coordinates, no market columns — and `score.py`
rejects the file if its columns change. So coordinates come from a city lookup
and `market_index` from the daily calendar (a national daily value: within-day
sd 0.025 against 0.168 overall). Lanes are represented by endpoint geometry —
midpoint, delta, bearing, haversine, circuity — rather than city or lane
identity, which also covers the 8 cities that appear only in validation.
`quote_signal` is excluded: it is absent from the December inputs entirely.

**Missing values are imputed, not dropped.** `weight` has 292 sign errors
(magnitudes are plausible; `abs()` repairs them) and 300 nulls; `market_index`
has 374 nulls. Both get an imputation flag so the model can condition on
missingness. Imputation statistics are fitted on development data only.

## Validation

The graded window is two months past the end of training, so every estimate
holds out a *trailing* window. Rolling origins at 2025-06-01, 07-01, 08-01 and
09-01, each with a 61-day horizon to match Nov–Dec. A random K-fold would let
the model see October while predicting June and would report a number the
graded set cannot reproduce.

Optuna tunes against the June/July/August origins only; the 2025-09-01 holdout
is never shown to the search, so the headline number is reproducible rather
than a training score.

Metrics are reported twice. `all_` covers every row; `clean_` excludes the
corrupted labels, which are unpredictable by construction and put a floor under
MAPE that no model can cross. Quoting only `all_` hides real improvement behind
that floor; quoting only `clean_` flatters the model against a grader that
scores all 12,000 rows.

## Results

Shipped model: **CatBoost hybrid on `coords`, \$/mile target, 50 Optuna
trials.** Evaluated on the withheld 2025-09-01 holdout — 9,523 loads spanning
2025-09-01 to 2025-10-31:

| Metric | All rows | Clean rows only |
|---|---|---|
| MAPE | **4.653%** | **2.163%** |
| MAE | **\$107.17** | **\$51.28** |
| RMSE | \$633.95 | \$76.37 |
| Median APE | 1.835% | 1.807% |
| R² | 0.8274 | 0.9969 |
| Rows | 9,523 | 9,379 |

Bias on all rows is −0.30%. Rolling-origin CV across the four folds is 4.430%
(sd 0.229).

`score.py` grades all 12,000 validation rows, so **4.653% is what predicts the
graded score**; the 2.163% is how well the model fits learnable signal. The gap
between them is the corruption floor, not a modelling choice — the 677 gated
labels carry a multiplier unrelated to any feature, so a perfect model still
scores 70–200% error on each.

### Backend and shape comparison

Rolling-origin CV with baseline hyperparameters held fixed across backends
(`reports/benchmark.csv` has the full 16-cell grid):

| learner | shape | encoding | MAPE | sd | MAE | clean | sec |
|---|---|---|---|---|---|---|---|
| catboost | hybrid | coords | 4.495 | 0.270 | \$107.52 | 2.227 | 83 |
| catboost | hybrid | coords+city | 4.526 | 0.269 | \$108.45 | 2.261 | 107 |
| xgboost | hybrid | coords+city | 4.540 | 0.281 | \$108.44 | 2.267 | 22 |
| lightgbm | hybrid | coords | 4.588 | 0.326 | \$109.77 | 2.316 | 6 |
| catboost | boosted | coords | 4.638 | 0.468 | \$111.16 | 2.378 | 59 |
| lightgbm | boosted | coords | 4.754 | 0.489 | \$114.07 | 2.490 | 5 |

- **Shape dominates library.** Every hybrid configuration beats every plain
  booster. The spread across the four backends within a shape (0.096pp) is
  smaller than the gap between shapes (0.142pp) and smaller than one fold
  standard deviation.
- **CatBoost wins, at ~14× the cost** — 83s against 6s for LightGBM on the
  same CV, for 0.093pp. Defensible to swap on a latency budget, hence
  `--learner`.
- **Identity encoding is noise.** `coords+city` and `coords` trade places
  depending on the backend, and the difference sits inside one fold sd. The
  shipped model uses `coords` alone: lane geometry generalises to the 8 cities
  that appear only in validation.
- **Tuning is the smaller half.** Fifty trials moved CatBoost's CV from 4.495
  to 4.430 (0.065pp); the structural changes — \$/mile target, label gate,
  hybrid date effect — did the rest.

`reports/holdout_metrics.csv` holds the per-fold CV detail and
`reports/runs.csv` the full run history.

## Layout

```
src/config.py      paths, constants, tuning knobs
src/data.py        loading, city lookup, market calendar, imputation, label gate
src/features.py    geographic and calendar feature construction
src/learners.py    the four boosting backends at matched capacity + Optuna spaces
src/models.py      $/mile target, parametric date effect, hybrid model
src/validation.py  metrics and the rolling-origin split
src/benchmark.py   the model comparison
src/train.py       tune, evaluate, write the reports/ artifacts
src/predict.py     write validation_predictions.csv and the December chart

score.py           the provided scorer; writes scorer_results/candidate_december.png
eda_notebook.ipynb data exploration behind the Approach section
data/              train-test.csv, validation.csv, the prediction template and
                   december-chart-inputs.csv (filled in place by predict.py)
reports/           metrics, benchmark grid, best_params.json, feature importance
```

Running `eda_notebook.ipynb` needs the two notebook pins at the end of
`requirements.txt`; the four commands above do not.

`train.py` writes `reports/best_params.json` and `predict.py` reads it, so the
shipped fit is the configuration that was measured. If that file is missing,
`predict.py` says so and falls back to the baseline rather than quietly
producing a submission with parameters nobody evaluated.
