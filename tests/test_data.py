import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (DataValidationError, RatioFeatures, load_data, preprocess_data,
                      split_data, validate_schema)

FEATURES = ["mean_radius", "mean_area", "mean_perimeter", "mean_texture", "mean_concavity"]
POSITIVE = ["mean_radius", "mean_area", "mean_perimeter", "mean_texture"]


def _df():
    return pd.DataFrame({
        "mean_radius": [10.0, 10.0, 12.0, -1.0, 14.0, 15.0],
        "mean_area": [300.0, 300.0, 400.0, 500.0, "abc", 600.0],
        "mean_perimeter": [60.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "mean_texture": [15.0, 15.0, 16.0, 17.0, 18.0, 0.0],
        "mean_concavity": [0.0, 0.0, 0.1, 0.2, 0.3, 0.4],
        "target": [0, 0, 1, 1, 0, None],
    })


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")


def test_validate_schema_reports_missing_columns():
    with pytest.raises(DataValidationError, match="mean_radius"):
        validate_schema(pd.DataFrame({"x": [1]}), ["mean_radius"], "target")


def test_preprocess_cleans_duplicates_invalid_and_bad_formats():
    out = preprocess_data(_df(), FEATURES, "target", POSITIVE)
    assert len(out) == 4                               # 1 duplicate + 1 row w/o target removed
    assert out["target"].dtype.kind == "i"
    assert out["mean_radius"].isna().sum() == 1        # -1.0 -> NaN
    assert out["mean_area"].isna().sum() == 1          # "abc" -> NaN
    assert (out["mean_concavity"] == 0.0).sum() == 1   # zero is VALID for non-strictly-positive columns


def test_split_is_stratified_and_sized():
    df = pd.DataFrame({"a": np.arange(100.0), "target": [0] * 50 + [1] * 50})
    Xtr, Xte, ytr, yte = split_data(df, ["a"], "target", 0.2, 42)
    assert len(Xte) == 20 and len(Xtr) == 80
    assert yte.sum() == 10  # stratified


def test_ratio_features_enforces_order_and_names():
    X = pd.DataFrame({"mean_radius": [10.0], "mean_texture": [20.0], "mean_area": [300.0], "mean_perimeter": [60.0]})
    rf = RatioFeatures().fit(X)
    shuffled = X[["mean_perimeter", "mean_radius", "mean_area", "mean_texture"]]
    out = rf.transform(shuffled)
    assert list(out.columns)[:4] == list(X.columns)
    assert out["area_to_perimeter"].iloc[0] == pytest.approx(5.0)
    with pytest.raises(DataValidationError):
        rf.transform(X.drop(columns=["mean_area"]))
    with pytest.raises(DataValidationError):
        rf.transform(X.to_numpy())


def test_importing_modules_does_not_start_training(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = "import src.data, src.train, src.evaluate, src.predict, src.api"
    env = {**os.environ, "PYTHONPATH": str(root)}
    res = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert list(tmp_path.iterdir()) == []  # nothing trained / written on import
