from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from . import config, features, learners, validation

warnings.filterwarnings("ignore")

SHAPES = ("boosted", "hybrid")

# Defined in features.py so the grid, train.py and predict.py cannot drift.
ENCODINGS = features.ENCODINGS


# Resolved by features.py so train.py and predict.py agree with the grid.
_columns_for = features.encoding_columns


def run(
    frame: pd.DataFrame,
    city_coords: pd.DataFrame,
    backends: list[str],
    shapes=SHAPES,
    encodings=tuple(ENCODINGS),
) -> pd.DataFrame:
    enriched = features.attach_identity(frame, city_coords)
    rows = []

    for encoding in encodings:
        for shape in shapes:
            columns, categorical = _columns_for(shape, encoding)
            source = enriched if ENCODINGS[encoding]["identity"] else frame
            for backend in backends:
                label = f"{backend:9s} {shape:8s} {encoding}"
                started = time.time()
                try:
                    cv = validation.cross_validate(
                        source,
                        kind=shape,
                        learner=backend,
                        feature_columns=columns,
                        categorical=categorical,
                    )
                except Exception as exc:  # unsupported combination, reported as such
                    print(f"  {label}  -- unsupported: {str(exc).strip()[:70]}")
                    rows.append(
                        {
                            "learner": backend,
                            "shape": shape,
                            "encoding": encoding,
                            "unsupported": str(exc).strip()[:120],
                        }
                    )
                    continue

                elapsed = time.time() - started
                record = {
                    "learner": backend,
                    "shape": shape,
                    "encoding": encoding,
                    "n_features": len(columns),
                    "all_mape": cv["all_mape"].mean(),
                    "all_mape_sd": cv["all_mape"].std(),
                    "all_mae": cv["all_mae"].mean(),
                    "clean_mape": cv["clean_mape"].mean(),
                    "bias": cv["all_bias"].mean(),
                    "seconds": elapsed,
                    **{f"fold_{c[5:]}": v for c, v in zip(cv["cutoff"], cv["all_mape"])},
                }
                rows.append(record)
                print(
                    f"  {label}  MAPE {record['all_mape']:6.3f}% "
                    f"(sd {record['all_mape_sd']:5.3f})  MAE ${record['all_mae']:7.2f}  "
                    f"clean {record['clean_mape']:6.3f}%  bias {record['bias']:+6.2f}%  "
                    f"{elapsed:5.1f}s"
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="*", default=list(SHAPES))
    parser.add_argument("--encodings", nargs="*", default=list(ENCODINGS))
    parser.add_argument("--learners", nargs="*", default=None)
    args = parser.parse_args()

    print("=" * 78)
    print("LEARNER / SHAPE / ENCODING BENCHMARK")
    print("=" * 78)

    backends = args.learners or learners.available()
    print(f"\nBackends that import and train here: {', '.join(backends)}")

    frame, city_coords = validation.load_development()
    print(f"Development rows: {len(frame):,}  corrupt labels gated from fits: "
          f"{int(frame['is_corrupt'].sum()):,}")
    print(f"Rolling origins: {', '.join(config.CV_CUTOFFS)} "
          f"(+{config.CV_HORIZON_DAYS}d horizon each)\n")

    table = run(frame, city_coords, backends, tuple(args.shapes), tuple(args.encodings))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "benchmark.csv"
    scored = table.dropna(subset=["all_mape"]) if "all_mape" in table else table
    table.sort_values("all_mape", na_position="last").to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("RANKED (mean all-row MAPE across folds)")
    print("=" * 78)
    cols = ["learner", "shape", "encoding", "all_mape", "all_mape_sd", "all_mae", "clean_mape",
            "bias", "seconds"]
    print(scored.sort_values("all_mape")[cols].to_string(index=False, float_format="%.3f"))

    best = scored.sort_values("all_mape").iloc[0]
    print(f"\nBest: {best.learner} / {best['shape']} / {best.encoding} "
          f"-> {best.all_mape:.3f}% MAPE")
    print(f"Saved {out}")
    print("Set config.LEARNER and config.MODEL to the winner, then run `python -m src.train`.")


if __name__ == "__main__":
    main()
