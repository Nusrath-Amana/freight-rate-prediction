"""Loading and cleaning.

1. 'weight' carries 292 negative values whose magnitudes are plausible --
   only the sign is wrong & 300 nulls.
2. 'market_index' has 374 nulls, fill from the daily calendar recovers it almost exactly.
3. 'posted_rate' carries 677 multiplicatively corrupted labels. These are
   dropped from training data

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def build_city_coords(*frames: pd.DataFrame) -> pd.DataFrame:
    # Map city name -> (lat, lon)
    parts = []
    for frame in frames:
        for role in ("pickup", "delivery"):
            cols = [role, f"{role}_lat", f"{role}_lon"]
            if all(col in frame.columns for col in cols):
                parts.append(
                    frame[cols].rename(
                        columns={role: "city", f"{role}_lat": "lat", f"{role}_lon": "lon"}
                    )
                )
    if not parts:
        raise ValueError("no coordinate columns found in the supplied frames")
    table = pd.concat(parts, ignore_index=True).dropna().drop_duplicates("city")
    return table.set_index("city").sort_index()


def build_market_calendar(*frames: pd.DataFrame) -> pd.Series:
    # Map date -> daily 'market_index'.

    parts = [f[["date", "market_index"]] for f in frames if "market_index" in f.columns]
    if not parts:
        raise ValueError("no market_index column found in the supplied frames")
    stacked = pd.concat(parts, ignore_index=True).dropna()
    return stacked.groupby("date")["market_index"].mean().sort_index()


class Cleaner:
    # Fits imputation statistics on the development data
    def __init__(self) -> None:
        self.weight_by_equipment: pd.Series | None = None
        self.weight_overall: float | None = None
        self.market_calendar: pd.Series | None = None
        self.peer_rpm: pd.Series | None = None
        self.distance_edges: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, market_calendar: pd.Series) -> "Cleaner":
        weight = _repair_weight_sign(train["weight"])
        self.weight_by_equipment = weight.groupby(train["equipment"]).median()
        self.weight_overall = float(weight.median())
        self.market_calendar = market_calendar

        # Peer-group posted_rate/distance medians, used only to gate corrupted labels.
        rpm = train["posted_rate"] / train["distance"]
        _, self.distance_edges = pd.qcut(
            train["distance"], config.DISTANCE_BINS, retbins=True, labels=False, duplicates="drop"
        )
        bucket = self._distance_bucket(train["distance"])
        self.peer_rpm = rpm.groupby([bucket, train["equipment"]]).median()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()

        out["weight"] = _repair_weight_sign(out["weight"])
        fallback = out["equipment"].map(self.weight_by_equipment).fillna(self.weight_overall)
        out["weight_imputed"] = out["weight"].isna().astype("int8")
        out["weight"] = out["weight"].fillna(fallback)

        from_calendar = out["date"].map(self.market_calendar)
        if "market_index" in out.columns:
            out["market_index_imputed"] = out["market_index"].isna().astype("int8")
            out["market_index"] = out["market_index"].fillna(from_calendar)
        else:
            out["market_index_imputed"] = np.int8(1)
            out["market_index"] = from_calendar
        out["market_index"] = out["market_index"].fillna(float(self.market_calendar.mean()))

        return out

    def flag_corrupt_labels(self, frame: pd.DataFrame) -> pd.Series:
        """True where 'posted_rate' looks multiplicatively corrupted."""
        rpm = frame["posted_rate"] / frame["distance"]
        bucket = self._distance_bucket(frame["distance"])
        index = pd.MultiIndex.from_arrays([bucket, frame["equipment"]])
        expected = pd.Series(self.peer_rpm.reindex(index).to_numpy(), index=frame.index)
        ratio = rpm / expected
        return ~ratio.between(config.RPM_RATIO_LOW, config.RPM_RATIO_HIGH) | ratio.isna()

    def _distance_bucket(self, distance: pd.Series) -> pd.Series:
        inner = self.distance_edges[1:-1]
        return pd.Series(np.digitize(distance.to_numpy(), inner), index=distance.index)


def _repair_weight_sign(weight: pd.Series) -> pd.Series:
    #Negative weights are sign errors, the magnitudes are in a valid range.
    return weight.abs()


def prepare(frames: dict[str, pd.DataFrame] | None = None):
    # Load and clean the development, validation, and December frames
    dev = load_raw(config.RAW_PATH)
    val = load_raw(config.VALIDATION_PATH)
    dec = load_raw(config.DECEMBER_PATH)

    city_coords = build_city_coords(dev, val)
    calendar = build_market_calendar(dev, val)

    cleaner = Cleaner().fit(dev, calendar)
    dev = cleaner.transform(dev)
    dev["is_corrupt"] = cleaner.flag_corrupt_labels(dev)
    val = cleaner.transform(val)
    dec = cleaner.transform(dec)

    return dev, val, dec, cleaner, city_coords
