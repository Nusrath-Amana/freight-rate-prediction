#One construction point for the four boosting backends, plus Optuna spaces.

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

LEARNERS = ("sklearn", "lightgbm", "xgboost", "catboost")


def available() -> list[str]:
    # smoke test
    probe_x = np.arange(300, dtype=float).reshape(100, 3)
    probe_y = np.arange(100, dtype=float)
    found = []
    for name in LEARNERS:
        try:
            regressor = make_regressor(name)
            regressor.set_params(**_tiny(name))
            regressor.fit(probe_x, probe_y)
            found.append(name)
        except Exception:
            continue
    return found


def _tiny(kind: str) -> dict:
    if kind == "sklearn":
        return {"max_iter": 2}
    if kind == "catboost":
        return {"iterations": 2}
    return {"n_estimators": 2}


def make_regressor(kind: str, categorical: list[str] | None = None, **overrides):
    # Return a new regressor of the requested backend, with the given categorical columns and any parameter overrides. The categorical list is only used for backends that need it; others ignore it.
    categorical = categorical or []

    if kind == "sklearn":
        from sklearn.ensemble import HistGradientBoostingRegressor

        params = dict(
            loss="squared_error",
            max_iter=config.N_ROUNDS,
            learning_rate=config.LEARNING_RATE,
            max_leaf_nodes=config.LEAVES,
            min_samples_leaf=config.MIN_LEAF,
            l2_regularization=config.L2,
            early_stopping=False,
            random_state=config.SEED,
        )
        params.update(overrides)
        return HistGradientBoostingRegressor(
            categorical_features=categorical or None, **params
        )

    if kind == "lightgbm":
        params = dict(
            objective="regression",
            num_boost_round=config.N_ROUNDS,
            learning_rate=config.LEARNING_RATE,
            num_leaves=config.LEAVES,
            min_data_in_leaf=config.MIN_LEAF,
            lambda_l2=config.L2,
            seed=config.SEED,
            verbose=-1,
        )
        params.update(overrides)
        return NativeLGBM(**params)

    if kind == "xgboost":
        from xgboost import XGBRegressor

        params = dict(
            n_estimators=config.N_ROUNDS,
            learning_rate=config.LEARNING_RATE,
            grow_policy="lossguide",  # leaf-wise, to match the others
            max_leaves=config.LEAVES,
            max_depth=0,  # unlimited; capacity governed by max_leaves
            min_child_weight=config.MIN_LEAF,
            reg_lambda=config.L2,
            random_state=config.SEED,
            enable_categorical=True,
            tree_method="hist",
            verbosity=0,
        )
        params.update(overrides)
        return XGBRegressor(**params)

    if kind == "catboost":
        from catboost import CatBoostRegressor

        params = dict(
            iterations=config.N_ROUNDS,
            learning_rate=config.LEARNING_RATE,
            depth=5,  # oblivious trees: 2**5 = 32 leaves
            min_data_in_leaf=config.MIN_LEAF,
            l2_leaf_reg=config.L2,
            random_seed=config.SEED,
            allow_writing_files=False,
            verbose=False,
        )
        params.update(overrides)
        return CatBoostRegressor(cat_features=categorical or None, **params)

    raise ValueError(f"unknown learner: {kind!r} (expected one of {LEARNERS})")


def suggest(kind: str, trial) -> dict:
    # optuna parmaeter ranges.
    rate = trial.suggest_float("learning_rate", 0.01, 0.12, log=True)
    rounds = trial.suggest_int("n_rounds", 300, 1400, step=100)
    leaves = trial.suggest_int("leaves", 16, 128, log=True)
    min_leaf = trial.suggest_int("min_leaf", 10, 120, log=True)
    l2 = trial.suggest_float("l2", 0.1, 20.0, log=True)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample = trial.suggest_float("colsample", 0.6, 1.0)

    if kind == "sklearn":
        # sklearn's HistGradientBoostingRegressor has no subsample or colsample parameters, so we ignore those.
        return dict(
            learning_rate=rate,
            max_iter=rounds,
            max_leaf_nodes=leaves,
            min_samples_leaf=min_leaf,
            l2_regularization=l2,
            max_features=colsample,
        )
    if kind == "lightgbm":
        return dict(
            learning_rate=rate,
            num_boost_round=rounds,
            num_leaves=leaves,
            min_data_in_leaf=min_leaf,
            lambda_l2=l2,
            bagging_fraction=subsample,
            bagging_freq=1,
            feature_fraction=colsample,
        )
    if kind == "xgboost":
        return dict(
            learning_rate=rate,
            n_estimators=rounds,
            max_leaves=leaves,
            min_child_weight=min_leaf,
            reg_lambda=l2,
            subsample=subsample,
            colsample_bytree=colsample,
        )
    if kind == "catboost":
        # CatBoost grows symmetric trees, so depth is the capacity knob;
        # 2**depth is the leaf count
        return dict(
            learning_rate=rate,
            iterations=rounds,
            depth=int(round(np.log2(leaves))),
            min_data_in_leaf=min_leaf,
            l2_leaf_reg=l2,
            rsm=colsample,
        )
    raise ValueError(f"unknown learner: {kind!r}")


def prepare_matrix(kind: str, matrix, categorical: list[str] | None = None):
    # LightGBM and XGBoost can handle categorical columns natively, but CatBoost requires them to be strings or objects. This function ensures that the categorical columns are in the correct format for the specified backend.
    categorical = [c for c in (categorical or []) if c in matrix.columns]
    if not categorical:
        return matrix
    out = matrix.copy()
    for column in categorical:
        if kind == "catboost":
            # CatBoost hashes raw labels; NaN is not an allowed category value.
            out[column] = out[column].astype("object").where(out[column].notna(), "__missing__")
            out[column] = out[column].astype(str)
        else:
            if not isinstance(out[column].dtype, pd.CategoricalDtype):
                out[column] = out[column].astype("category")
    return out


class NativeLGBM:
    # A wrapper around LightGBM's native API to provide a scikit-learn-like interface. This allows us to use LightGBM with the same fit/predict methods as the other backends, while still taking advantage of LightGBM's native features.

    def __init__(self, **params):
        self.params = params
        self.booster_ = None
        self._categorical: list[str] = []

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, x, y, **_):
        import lightgbm as lgb

        frame = x.copy() if isinstance(x, pd.DataFrame) else pd.DataFrame(x).copy()
        self._categorical = [
            c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)
        ]
        params = dict(self.params)
        rounds = int(
            params.pop("num_boost_round", params.pop("n_estimators", config.N_ROUNDS))
        )
        dataset = lgb.Dataset(
            frame,
            label=np.ascontiguousarray(np.asarray(y, dtype=float)),
            categorical_feature=self._categorical or "auto",
            free_raw_data=False,
            params=params,
        )
        dataset.construct()
        self.booster_ = lgb.train(params, dataset, num_boost_round=rounds)
        return self

    def predict(self, x):
        frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        return self.booster_.predict(frame)

    @property
    def feature_importances_(self):
        return self.booster_.feature_importance(importance_type="gain")
