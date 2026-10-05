
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features, learners


class RateModel:
    """A single booster on the posted_rate/distance target."""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        learner: str | None = None,
        categorical: list[str] | None = None,
        params: dict | None = None,
    ) -> None:
        self.feature_columns = list(feature_columns or features.FEATURES)
        self.learner = learner or config.LEARNER
        cats = categorical if categorical is not None else features.CATEGORICAL
        self.categorical = [c for c in cats if c in self.feature_columns]
        self.params = dict(params or {})
        self.regressor = learners.make_regressor(self.learner, self.categorical, **self.params)

    def _matrix(self, frame: pd.DataFrame):
        return learners.prepare_matrix(
            self.learner, features.matrix(frame, self.feature_columns), self.categorical
        )

    def fit(self, frame: pd.DataFrame) -> "RateModel":
        self.regressor.fit(
            self._matrix(frame), frame["posted_rate"] / frame["distance"]
        )
        return self

    def predict_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(self._matrix(frame))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_rpm(frame) * frame["distance"].to_numpy()

    def importances(self) -> pd.Series | None:
        raw = getattr(self.regressor, "feature_importances_", None)
        if raw is None:
            return None
        return pd.Series(np.asarray(raw, dtype=float), index=self.feature_columns).sort_values(
            ascending=False
        )


class DateEffect:

    def __init__(self) -> None:
        self.origin: pd.Timestamp | None = None
        self.coefficients: pd.Series | None = None

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        design = pd.DataFrame(index=frame.index)
        dow = frame.index.dayofweek
        for day in range(1, 7):
            design[f"dow_{day}"] = (dow == day).astype(float)
        design["market_index"] = frame["market_index"].to_numpy()
        design["elapsed"] = (frame.index - self.origin).days / 100.0
        design["intercept"] = 1.0
        return design

    @staticmethod
    def _daily(frame: pd.DataFrame, effect: np.ndarray) -> pd.DataFrame:
        stacked = pd.DataFrame(
            {
                "date": frame["date"].to_numpy(),
                "effect": effect,
                "market_index": frame["market_index"].to_numpy(),
            }
        )
        return stacked.groupby("date").mean().sort_index()

    def fit(self, frame: pd.DataFrame, effect: np.ndarray) -> "DateEffect":
        daily = self._daily(frame, effect)
        self.origin = daily.index.min()
        design = self._design(daily)
        solution, *_ = np.linalg.lstsq(
            design.to_numpy(), daily["effect"].to_numpy(), rcond=None
        )
        self.coefficients = pd.Series(solution, index=design.columns)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        daily_index = pd.DatetimeIndex(frame["date"])
        design = self._design(
            pd.DataFrame({"market_index": frame["market_index"].to_numpy()}, index=daily_index)
        )
        return design.to_numpy() @ self.coefficients.reindex(design.columns).to_numpy()


class HybridRateModel:
    # Boosted cross-sectional structure plus an extrapolating date effect.

    def __init__(
        self,
        n_backfit: int = 1,
        learner: str | None = None,
        feature_columns: list[str] | None = None,
        categorical: list[str] | None = None,
        params: dict | None = None,
    ) -> None:
        self.feature_columns = list(feature_columns or features.STRUCTURAL_FEATURES)
        self.n_backfit = n_backfit
        self.learner = learner or config.LEARNER
        self.categorical = categorical
        self.params = dict(params or {})
        self.structural = None
        self.date_effect = DateEffect()

    def _cats(self) -> list[str]:
        cats = self.categorical if self.categorical is not None else features.CATEGORICAL
        return [c for c in cats if c in self.feature_columns]

    def _matrix(self, frame: pd.DataFrame):
        return learners.prepare_matrix(
            self.learner, features.matrix(frame, self.feature_columns), self._cats()
        )

    def fit(self, frame: pd.DataFrame) -> "HybridRateModel":
        y = (frame["posted_rate"] / frame["distance"]).to_numpy()
        x = self._matrix(frame)
        offset = np.zeros_like(y)
        for _ in range(self.n_backfit + 1):
            self.structural = learners.make_regressor(
                self.learner, self._cats(), **self.params
            )
            self.structural.fit(x, y - offset)
            self.date_effect.fit(frame, y - self.structural.predict(x))
            offset = self.date_effect.predict(frame)
        return self

    def predict_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        structural = self.structural.predict(self._matrix(frame))
        return structural + self.date_effect.predict(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_rpm(frame) * frame["distance"].to_numpy()

    def importances(self) -> pd.Series | None:
        raw = getattr(self.structural, "feature_importances_", None)
        if raw is None:
            return None
        return pd.Series(np.asarray(raw, dtype=float), index=self.feature_columns).sort_values(
            ascending=False
        )


def trend_collinearity(frame: pd.DataFrame) -> float:
    # Compute the correlation between date and the market index. A high correlation indicates that the market index is trending strongly over time, which can cause collinearity issues when fitting a hybrid model that includes both structural features and a date effect.
    daily = frame.groupby("date")["market_index"].mean().sort_index()
    if len(daily) < 3:
        return 1.0
    elapsed = (daily.index - daily.index.min()).days.to_numpy(dtype=float)
    return float(np.corrcoef(elapsed, daily.to_numpy())[0, 1])


def build(frame: pd.DataFrame | None = None, kind: str | None = None, **overrides):
    # Pick the model, falling back to the booster when the drift is unidentified.

    kind = kind or config.MODEL
    if kind == "hybrid" and frame is not None:
        if abs(trend_collinearity(frame)) > config.MAX_TREND_COLLINEARITY:
            kind = "boosted"
            columns = list(overrides.get("feature_columns") or features.STRUCTURAL_FEATURES)
            overrides["feature_columns"] = columns + [
                c for c in features.FEATURES if c not in columns
            ]
    return HybridRateModel(**overrides) if kind == "hybrid" else RateModel(**overrides)
