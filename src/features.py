
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EQUIPMENT_LEVELS = ("Dry Van", "Flatbed", "Reefer")

BASE_FEATURES = [
    "distance",
    "weight",
    "weight_imputed",
    "equipment_code",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "mid_lat",
    "mid_lon",
    "delta_lat",
    "delta_lon",
    "bearing_sin",
    "bearing_cos",
    "haversine_mi",
    "circuity",
    "is_short_haul"
]
MARKET_FEATURES = ["market_index", "market_index_imputed"]

CATEGORICAL = ["equipment_code"]

CALENDAR_FEATURES = ["dow", "is_weekend", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]
FEATURES = BASE_FEATURES + MARKET_FEATURES + CALENDAR_FEATURES

# hybrid model only sees the structural features, not the market_index or calendar features, which are handled by the parametric date effect.
STRUCTURAL_FEATURES = list(BASE_FEATURES)


def haversine_miles(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2 - lon1)
    inner = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    return 2.0 * config.EARTH_RADIUS_MI * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def attach_coordinates(frame: pd.DataFrame, city_coords: pd.DataFrame) -> pd.DataFrame:
    
    out = frame.copy()
    for role in ("pickup", "delivery"):
        looked_up = out[role].map(city_coords["lat"]), out[role].map(city_coords["lon"])
        for col, values in zip((f"{role}_lat", f"{role}_lon"), looked_up):
            out[col] = out[col].fillna(values) if col in out.columns else values
    missing = out[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].isna().any(axis=1)
    if missing.any():
        unknown = sorted(set(out.loc[missing, "pickup"]) | set(out.loc[missing, "delivery"]))
        raise ValueError(f"cities absent from the coordinate lookup: {unknown}")
    return out


def build(frame: pd.DataFrame, city_coords: pd.DataFrame) -> pd.DataFrame:
    out = attach_coordinates(frame, city_coords)

    out["is_short_haul"] = (out["distance"] <= config.DISTANCE_FLOOR_MI).astype("int8")

    out["equipment_code"] = pd.Categorical(
        out["equipment"], categories=EQUIPMENT_LEVELS
    ).codes.astype("int8")

    out["mid_lat"] = (out["pickup_lat"] + out["delivery_lat"]) / 2.0
    out["mid_lon"] = (out["pickup_lon"] + out["delivery_lon"]) / 2.0
    out["delta_lat"] = out["delivery_lat"] - out["pickup_lat"]
    out["delta_lon"] = out["delivery_lon"] - out["pickup_lon"]
    bearing = np.arctan2(out["delta_lon"], out["delta_lat"])
    out["bearing_sin"] = np.sin(bearing)
    out["bearing_cos"] = np.cos(bearing)

    out["haversine_mi"] = haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    
    out["circuity"] = out["distance"] / out["haversine_mi"].clip(lower=1e-6)

    out["dow"] = out["date"].dt.dayofweek.astype("int8")
    out["is_weekend"] = out["dow"].isin((5, 6)).astype("int8")
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    doy = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)

    return out


def matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    return frame[list(columns or FEATURES)]



IDENTITY_FEATURES = ["pickup_cat", "delivery_cat"]


ENCODINGS = {
    "coords": {"columns": [], "categorical": [], "identity": False},
    "coords+city": {
        "columns": ["pickup_cat", "delivery_cat"],
        "categorical": ["pickup_cat", "delivery_cat"],
        "identity": True,
    },
}


def encoding_columns(shape: str, encoding: str) -> tuple[list[str], list[str]]:
    """Feature and categorical lists for a (shape, encoding) pair."""
    base = STRUCTURAL_FEATURES if shape == "hybrid" else FEATURES
    spec = ENCODINGS[encoding]
    return (list(base) + list(spec["columns"]),
            list(CATEGORICAL) + list(spec["categorical"]))


def apply_encoding(frame: pd.DataFrame, city_coords: pd.DataFrame,
                   encoding: str) -> pd.DataFrame:
    """Attach whatever identity columns the encoding needs; a no-op for coords."""
    return attach_identity(frame, city_coords) if ENCODINGS[encoding]["identity"] else frame


def attach_identity(frame: pd.DataFrame, city_coords: pd.DataFrame) -> pd.DataFrame:
    
    cities = list(city_coords.index)
    out = frame.copy()
    out["pickup_cat"] = pd.Categorical(out["pickup"], categories=cities)
    out["delivery_cat"] = pd.Categorical(out["delivery"], categories=cities)
    return out
