"""Data stage: load, validate, clean, feature-engineer and split.

Importing this module has no side effects (nothing runs at import time).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when data does not match the expected schema."""


# --------------------------------------------------------------------------- #
# Public data source (approved: scikit-learn's bundled Breast Cancer Wisconsin)
# --------------------------------------------------------------------------- #
def export_breast_cancer_csv(path: str | Path, features: Sequence[str] | None = None) -> Path:
    """Write the public scikit-learn breast-cancer dataset to ``path`` as CSV.

    Column names are converted to snake_case ("mean concave points" -> "mean_concave_points").
    """
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    if features is not None:
        df = df[[*features, "target"]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Stage 1: loading
# --------------------------------------------------------------------------- #
def load_data(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV. Fails loudly when the file is missing or empty."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run `dvc pull` or `python scripts/make_dataset.py`."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise DataValidationError(f"Dataset '{path}' is empty.")
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return df


def validate_schema(df: pd.DataFrame, features: Sequence[str], target: str) -> None:
    """Check that every expected column is present."""
    missing = [c for c in [*features, target] if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing expected columns: {missing}")


# --------------------------------------------------------------------------- #
# Stage 2: cleaning
# --------------------------------------------------------------------------- #
def preprocess_data(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    strictly_positive: Iterable[str] = (),
) -> pd.DataFrame:
    """Clean raw data.

    * keep only the expected columns (consistent column names / order)
    * drop exact duplicate rows and rows without a target
    * coerce features to numeric (bad formats become NaN)
    * mark invalid values (negative, or <= 0 for ``strictly_positive`` columns) as NaN

    Missing/invalid feature values are NOT imputed here. Imputation is a *learned*
    transformation, so it lives inside the saved sklearn Pipeline and is fitted on the
    training split only (prevents test-set leakage).
    """
    validate_schema(df, features, target)
    out = df[[*features, target]].copy()
    n_start = len(out)

    out = out.drop_duplicates()
    n_dups = n_start - len(out)

    out = out.dropna(subset=[target])
    n_no_target = n_start - n_dups - len(out)

    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    strictly_positive = list(strictly_positive)
    invalid = out[list(features)] < 0
    for col in strictly_positive:
        invalid[col] = out[col] <= 0
    n_invalid = int(invalid.to_numpy().sum())
    out[list(features)] = out[list(features)].mask(invalid)

    out[target] = out[target].astype(int)
    logger.info(
        "Cleaning: %d duplicates dropped, %d rows without target dropped, %d invalid values set to NaN",
        n_dups, n_no_target, n_invalid,
    )
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 3: feature engineering (a transformer so it ships inside the Pipeline)
# --------------------------------------------------------------------------- #
class RatioFeatures(BaseEstimator, TransformerMixin):
    """Adds two ratio features and enforces the exact training column set/order."""

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.columns_ = list(X.columns)
        self.n_features_in_ = len(self.columns_)
        return self

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise DataValidationError("RatioFeatures expects a pandas DataFrame with named columns.")
        return X

    def transform(self, X):
        X = self._as_frame(X)
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise DataValidationError(f"Missing required feature columns: {missing}")
        X = X[self.columns_].copy()  # preserves training order regardless of input order
        if {"mean_area", "mean_perimeter"} <= set(X.columns):
            X["area_to_perimeter"] = X["mean_area"] / X["mean_perimeter"]
        if {"mean_radius", "mean_texture"} <= set(X.columns):
            X["radius_to_texture"] = X["mean_radius"] / X["mean_texture"]
        return X.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Stage 4: splitting
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    test_size: float,
    random_state: int,
):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X, y = df[list(features)], df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
