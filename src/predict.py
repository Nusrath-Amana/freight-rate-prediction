from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from . import config, data, features, models

warnings.filterwarnings("ignore")


def load_best() -> tuple[str, str, dict]:

    if not config.BEST_PARAMS_PATH.exists():
        print(f"  ! {config.BEST_PARAMS_PATH.name} not found; using baseline parameters.")
        print("    Run `python -m src.train` first for the tuned fit.")
        return config.LEARNER, config.MODEL, {}
    best = json.loads(config.BEST_PARAMS_PATH.read_text())
    print(f"  loaded {config.BEST_PARAMS_PATH.name}: {best['learner']} / {best['shape']}, "
          f"tuned={best['tuned']} over {best['trials']} trials")
    print(f"    measured holdout MAPE {best['holdout_mape']}% "
          f"(clean {best['holdout_clean_mape']}%), CV {best['cv_mape']}%")
    return (best["learner"], best["shape"], best.get("encoding", config.ENCODING),
            best.get("native_params") or {})


def main() -> None:
    print("=" * 78)
    print("PREDICT")
    print("=" * 78)

    learner, shape, encoding, params = load_best()

    dev, val, dec, cleaner, city_coords = data.prepare()
    dev = features.apply_encoding(features.build(dev, city_coords), city_coords, encoding)
    fitting = dev[~dev["is_corrupt"]]

    collinearity = models.trend_collinearity(dev)
    if shape == "hybrid" and abs(collinearity) > config.MAX_TREND_COLLINEARITY:
        print(f"  trend collinearity {collinearity:+.3f} -> falling back to plain booster")
        shape = "boosted"

    print(f"\nFitting {learner} / {shape} / {encoding} on {len(fitting):,} clean rows "
          f"({dev['date'].min().date()} .. {dev['date'].max().date()})")
    columns, categorical = features.encoding_columns(shape, encoding)
    model = models.build(fitting, kind=shape, learner=learner, params=params,
                         feature_columns=columns, categorical=categorical)
    model.fit(fitting)

    # --- validation.csv ---------------------------------------------------
    val_features = features.apply_encoding(features.build(val, city_coords),
                                           city_coords, encoding)
    val_pred = model.predict(val_features)
    template = data.load_raw(config.TEMPLATE_PATH)
    predictions = pd.Series(val_pred, index=val_features["load_id"].to_numpy())
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(predictions).round(2)

    missing = int(out["predicted_rate"].isna().sum())
    if missing:
        raise ValueError(f"{missing} template load_ids have no prediction")
    out.to_csv(config.PREDICTIONS_OUT, index=False)
    print(f"\nWrote {config.PREDICTIONS_OUT.name}  ({len(out):,} rows)")
    print(f"  predicted rate   mean ${out.predicted_rate.mean():.2f}  "
          f"min ${out.predicted_rate.min():.2f}  max ${out.predicted_rate.max():.2f}")
    print(f"  implied $/mile   mean "
          f"{(out.predicted_rate.to_numpy() / val_features.distance.to_numpy()).mean():.3f}")

    # --- december chart ---------------------------------------------------
    dec_features = features.apply_encoding(features.build(dec, city_coords),
                                           city_coords, encoding)
    dec_pred = model.predict(dec_features)
    raw_dec = pd.read_csv(config.DECEMBER_PATH)
    raw_dec["predicted_rate"] = np.round(dec_pred, 2)
    raw_dec.to_csv(config.DECEMBER_PATH, index=False)
    print(f"\nFilled {config.DECEMBER_PATH.name}  ({len(raw_dec)} rows)")
    print(f"  Dec 1 ${raw_dec.predicted_rate.iloc[0]:.2f} .. "
          f"Dec 31 ${raw_dec.predicted_rate.iloc[-1]:.2f}  "
          f"(range ${raw_dec.predicted_rate.min():.2f}-${raw_dec.predicted_rate.max():.2f})")



if __name__ == "__main__":
    main()
