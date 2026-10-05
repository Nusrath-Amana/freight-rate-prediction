
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, features, models


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    ape = np.abs(error) / actual
    return {
        "mape": float(np.mean(ape) * 100.0),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_ape": float(np.median(ape) * 100.0),
        "bias": float(np.mean(predicted / actual - 1.0) * 100.0),
        "r2": float(1.0 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
        "n": int(len(actual)),
    }


def evaluate(test: pd.DataFrame, predicted: np.ndarray) -> dict:
    out = {f"all_{k}": v for k, v in metrics(test["posted_rate"].to_numpy(), predicted).items()}
    if "is_corrupt" in test.columns:
        clean = ~test["is_corrupt"].to_numpy()
        out.update(
            {
                f"clean_{k}": v
                for k, v in metrics(
                    test["posted_rate"].to_numpy()[clean], predicted[clean]
                ).items()
            }
        )
    return out


def folds(frame: pd.DataFrame, cutoffs=None, horizon_days: int | None = None):
    cutoffs = cutoffs or config.CV_CUTOFFS
    horizon = pd.Timedelta(days=horizon_days or config.CV_HORIZON_DAYS)
    for cutoff in cutoffs:
        start = pd.Timestamp(cutoff)
        train = frame[frame["date"] < start]
        test = frame[(frame["date"] >= start) & (frame["date"] < start + horizon)]
        if len(train) and len(test):
            yield cutoff, train, test


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, **build_kwargs):
    fitting = train[~train["is_corrupt"]] if "is_corrupt" in train.columns else train
    model = models.build(fitting, **build_kwargs)
    model.fit(fitting)
    return model, model.predict(test)


def cross_validate(frame: pd.DataFrame, **build_kwargs) -> pd.DataFrame:
    # Rolling-origin CV. One row per fold
    rows = []
    for cutoff, train, test in folds(frame):
        _, predicted = fit_predict(train, test, **build_kwargs)
        rows.append(
            {
                "cutoff": cutoff,
                "n_train": len(train),
                "n_test": len(test),
                **evaluate(test, predicted),
            }
        )
    return pd.DataFrame(rows)


def load_development() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev, val, _, _, city_coords = data.prepare()
    return features.build(dev, city_coords), city_coords
