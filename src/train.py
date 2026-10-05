from __future__ import annotations

import argparse
import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from . import config, data, features, learners, models, validation

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_CUTOFFS = config.CV_CUTOFFS[:-1]  


def _progress(total: int, started: float):
    """Optuna callback: one line per trial, with a projected finish time.

    Without this a long search is indistinguishable from a hung process, which
    is the difference between waiting another ten minutes and killing it.
    """
    def report(study, trial):
        done = len(study.trials)
        elapsed = time.time() - started
        rate = elapsed / max(done, 1)
        remaining = rate * (total - done)
        value = "pruned" if trial.value is None else f"{trial.value:6.3f}%"
        print(f"  trial {done:3d}/{total}  {value}   best {study.best_value:6.3f}%"
              f"   {elapsed/60:5.1f}m elapsed, ~{remaining/60:4.1f}m left",
              flush=True)
    return report


def objective(trial, frame: pd.DataFrame, learner: str, shape: str,
              columns: list[str], categorical: list[str]) -> float:
    params = learners.suggest(learner, trial)
    scores = []
    for _, train, test in validation.folds(frame, cutoffs=TUNING_CUTOFFS):
        try:
            _, predicted = validation.fit_predict(
                train, test, kind=shape, learner=learner, params=params,
                feature_columns=columns, categorical=categorical,
            )
        except Exception:
            raise optuna.TrialPruned()
        scores.append(validation.metrics(test["posted_rate"].to_numpy(), predicted)["mape"])
    return float(np.mean(scores))


def plot_importance(importances: pd.Series, path) -> None:
    top = importances.head(20)[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top.index, top.to_numpy(), color="#4C78A8")
    plt.xlabel("importance")
    plt.title("Feature importance (structural fit)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner", default=config.LEARNER)
    parser.add_argument("--shape", default=config.MODEL, choices=("boosted", "hybrid"))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--encoding", default=config.ENCODING,
                        choices=tuple(features.ENCODINGS),
                        help="lane representation; see benchmark.py")
    parser.add_argument("--no-tune", action="store_true", help="use baseline params only")
    args = parser.parse_args()

    print("=" * 78)
    print("TRAIN")
    print("=" * 78)

    frame, city_coords = validation.load_development()
    frame = features.apply_encoding(frame, city_coords, args.encoding)
    clean = int((~frame["is_corrupt"]).sum())
    print(f"\nDevelopment rows {len(frame):,}  usable for fitting {clean:,} "
          f"({int(frame['is_corrupt'].sum()):,} corrupt labels gated out)")

    collinearity = models.trend_collinearity(frame)
    shape = args.shape
    if shape == "hybrid" and abs(collinearity) > config.MAX_TREND_COLLINEARITY:
        print(f"  trend collinearity {collinearity:+.3f} exceeds "
              f"{config.MAX_TREND_COLLINEARITY}; falling back to the plain booster")
        shape = "boosted"
    else:
        print(f"  trend collinearity {collinearity:+.3f} -> drift term is identified")
    columns, categorical = features.encoding_columns(shape, args.encoding)
    print(f"  learner={args.learner}  shape={shape}  encoding={args.encoding}"
          f"  ({len(columns)} features)")

    params: dict = {}
    if not args.no_tune:
        print(f"\nTuning on {', '.join(TUNING_CUTOFFS)} ({args.trials} trials); "
              f"{config.HOLDOUT_START} holdout withheld from the search")
        study = optuna.create_study(direction="minimize", study_name=f"{args.learner}_{shape}")
        started = time.time()
        study.optimize(
            lambda t: objective(t, frame, args.learner, shape, columns, categorical),
            n_trials=args.trials,
            show_progress_bar=False,
            callbacks=[_progress(args.trials, started)],
        )
        print(f"  search finished in {(time.time() - started) / 60:.1f} minutes")
        print(f"  best tuning MAPE {study.best_value:.3f}%")
        print(f"  best params {study.best_params}")
        params = learners.suggest(args.learner, optuna.trial.FixedTrial(study.best_params))
    else:
        print("\nSkipping the search; using the matched baseline parameters.")

    # --- Honest forward holdout -------------------------------------------
    holdout_train = frame[frame["date"] < config.HOLDOUT_START]
    holdout_test = frame[frame["date"] >= config.HOLDOUT_START]
    print("\nScoring the withheld holdout...", flush=True)
    model, predicted = validation.fit_predict(
        holdout_train, holdout_test, kind=shape, learner=args.learner, params=params,
        feature_columns=columns, categorical=categorical,
    )
    scores = validation.evaluate(holdout_test, predicted)

    print(f"\n=== Holdout {config.HOLDOUT_START} .. "
          f"{holdout_test['date'].max().date()} ({scores['all_n']:,} rows) ===")
    print(f"MAPE        {scores['all_mape']:6.3f}%   (clean {scores['clean_mape']:6.3f}%)")
    print(f"MAE         ${scores['all_mae']:8.2f}   (clean ${scores['clean_mae']:7.2f})")
    print(f"RMSE        ${scores['all_rmse']:8.2f}")
    print(f"median APE  {scores['all_median_ape']:6.3f}%")
    print(f"bias        {scores['all_bias']:+6.2f}%")
    print(f"R2          {scores['all_r2']:6.4f}")

    print(f"\nRunning {len(config.CV_CUTOFFS)}-fold rolling-origin CV...", flush=True)
    cv = validation.cross_validate(frame, kind=shape, learner=args.learner, params=params,
                                   feature_columns=columns, categorical=categorical)
    print(f"\nRolling-origin CV mean MAPE {cv['all_mape'].mean():.3f}% "
          f"(sd {cv['all_mape'].std():.3f}) over {len(cv)} folds")

    # Refit on the full development window and persist
    print("\nRefitting on the full development window (Jan-Oct, clean rows)...",
          flush=True)
    final = models.build(
        frame[~frame["is_corrupt"]],
        kind=shape,
        learner=args.learner,
        params=params,
        feature_columns=columns,
        categorical=categorical,
    )
    final.fit(frame[~frame["is_corrupt"]])

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv.to_csv(config.REPORTS_DIR / "holdout_metrics.csv", index=False)

    config.BEST_PARAMS_PATH.write_text(
        json.dumps(
            {
                "learner": args.learner,
                "shape": shape,
                "encoding": args.encoding,
                "tuned": not args.no_tune,
                "trials": 0 if args.no_tune else args.trials,
                "search_space_params": {} if args.no_tune else study.best_params,
                "native_params": params,
                "tuning_mape": None if args.no_tune else round(study.best_value, 4),
                "holdout_mape": round(scores["all_mape"], 4),
                "holdout_clean_mape": round(scores["clean_mape"], 4),
                "cv_mape": round(float(cv["all_mape"].mean()), 4),
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nWrote {config.BEST_PARAMS_PATH.relative_to(config.ROOT)}")

    run_record = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "learner": args.learner,
        "shape": shape,
        "encoding": args.encoding,
        "tuned": not args.no_tune,
        "trials": 0 if args.no_tune else args.trials,
        "target": "posted_rate / distance",
        "n_features": len(final.feature_columns),
        "corrupt_labels_gated": int(frame["is_corrupt"].sum()),
        "trend_collinearity": round(collinearity, 4),
        "cv_mape": round(float(cv["all_mape"].mean()), 4),
        "cv_mape_sd": round(float(cv["all_mape"].std()), 4),
        **{f"holdout_{k}": round(float(v), 4) for k, v in scores.items()},
        **{f"hp_{k}": v for k, v in params.items()},
    }
    history = config.REPORTS_DIR / "runs.csv"
    pd.DataFrame([run_record]).to_csv(
        history, mode="a", header=not history.exists(), index=False
    )

    importances = final.importances()
    if importances is not None:
        plot_importance(importances, config.REPORTS_DIR / "feature_importance.png")
        importances.to_csv(config.REPORTS_DIR / "feature_importance.csv")
        print("\nTop features:")
        print(importances.head(10).to_string())

    print(f"\nRecorded under {config.REPORTS_DIR.name}/:")
    for name in ("runs.csv", "best_params.json", "holdout_metrics.csv",
                 "feature_importance.csv", "feature_importance.png"):
        print(f"  {name}")
    print("Then write predictions with `python -m src.predict`.")


if __name__ == "__main__":
    main()
