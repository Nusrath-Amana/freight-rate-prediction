"""Paths, constants and tuning knobs for the freight rate pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

RAW_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"

PREDICTIONS_OUT = ROOT / "validation_predictions.csv"

SEED = 42
EARTH_RADIUS_MI = 3958.7613

TARGET = "posted_rate"

# --- Data quality ------------------------------------------------------------
# REmove corrupted label rows.
# A row's posted_rate/distance divided by the median posted_rate/distance of its peer group (distance
# ventile x equipment). The observed distribution is bimodal with genuinely
# empty gaps at [0.50, 0.75] and [1.25, 2.00)
RPM_RATIO_LOW = 0.6
RPM_RATIO_HIGH = 1.5

DISTANCE_BINS = 20  # ventiles, for the peer-group outlier gate
DISTANCE_FLOOR_MI = 70.0  # observed hard floor in the source data

# --- Validation --------------------------------------------------------------
# Development data spans 2025-01-01 .. 2025-10-31; the graded set is Nov-Dec.
# Holding out the trailing months reproduces that forward-extrapolation gap.
HOLDOUT_START = "2025-09-01"
CV_CUTOFFS = ("2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01")
CV_HORIZON_DAYS = 61

# --- Model selection ---------------------------------------------------------
# "hybrid" adds a linear drift term to the booster
MODEL = "hybrid"

LEARNER = "xgboost"

ENCODING = "coords"

BEST_PARAMS_PATH = REPORTS_DIR / "best_params.json"

# hybrid model is backed by a single booster if the linear trend is collinear with the booster features.
MAX_TREND_COLLINEARITY = 0.55

# --- Baseline hyperparameters ------------------------------------------------
LEARNING_RATE = 0.05
N_ROUNDS = 600
LEAVES = 32
MIN_LEAF = 40
L2 = 1.0

