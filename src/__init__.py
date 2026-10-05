"""Spotter freight rate prediction pipeline.

LightGBM must be imported before pandas in this environment. Both ship their
own OpenMP runtime, and on Windows whichever loads second binds against the
first; with pandas 2.2.2 / LightGBM 4.7.0 / numpy 2.5.3 the result is an
access violation inside ``LGBM_DatasetSetField`` the moment a Dataset of any
real size is constructed. Measured here: importing pandas first crashes on a
47,323 x 17 array of random normals, while the reverse order trains the same
data fine. CatBoost triggers it too, by importing pandas itself.

It is a one-line fix in exactly one place, and it has to be this place --
every other module imports pandas at its top, so by the time ``learners.py``
reaches its lazy ``import lightgbm`` the damage is done. The import is guarded
because LightGBM is optional: the pipeline runs without it.
"""
try:  # noqa: SIM105 - must precede any pandas import in this package
    import lightgbm as _lightgbm  # noqa: F401
except Exception:  # pragma: no cover - LightGBM is an optional backend
    pass
