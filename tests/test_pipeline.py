import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError
from src.predict import feature_names, load_model, predict

ROOT = Path(__file__).resolve().parents[1]


def test_artifacts_created(artifacts):
    cfg = artifacts["cfg"]
    assert Path(cfg["paths"]["model_path"]).exists()
    assert Path(cfg["paths"]["metadata_path"]).exists()
    for f in ("metrics.json", "evaluation_report.md", "classification_report.txt",
              "confusion_matrix.png", "roc_curve.png"):
        assert (Path(cfg["paths"]["reports_dir"]) / f).exists(), f


def test_saved_object_is_full_pipeline(artifacts):
    model = load_model(artifacts["cfg"]["paths"]["model_path"])
    assert list(model.named_steps) == ["features", "imputer", "scaler", "model"]


def test_metrics_on_held_out_data_are_reasonable(artifacts):
    m = artifacts["metrics"]
    assert m["n_test_samples"] == 114
    assert m["accuracy"] > 0.90 and m["roc_auc"] > 0.95


def test_test_data_not_used_to_fit_preprocessing(artifacts):
    """Scaler statistics must equal the TRAIN split's statistics, not the full data's."""
    cfg = artifacts["cfg"]
    model = load_model(cfg["paths"]["model_path"])
    proc = Path(cfg["data"]["processed_dir"])
    train = pd.read_csv(proc / "train.csv")
    test = pd.read_csv(proc / "test.csv")
    feats = model.named_steps["features"].transform(train[feature_names(model)])
    assert np.allclose(model.named_steps["scaler"].mean_, feats.mean().to_numpy())
    full = pd.concat([train, test])
    full_feats = model.named_steps["features"].transform(full[feature_names(model)])
    assert not np.allclose(model.named_steps["scaler"].mean_, full_feats.mean().to_numpy())
    # train and test are disjoint
    assert set(map(tuple, train.round(8).to_numpy())).isdisjoint(set(map(tuple, test.round(8).to_numpy())))


def test_predict_output_shape_and_values(artifacts):
    cfg = artifacts["cfg"]
    model = load_model(cfg["paths"]["model_path"])
    rows = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "test.csv").head(7)
    out = predict(rows, model=model, class_names=cfg["data"]["class_names"])
    assert len(out) == 7
    for r in out:
        assert r["prediction"] in (0, 1)
        assert r["label"] in ("malignant", "benign")
        assert sum(r["probabilities"].values()) == pytest.approx(1.0)


def test_predict_rejects_missing_features(artifacts):
    model = load_model(artifacts["cfg"]["paths"]["model_path"])
    with pytest.raises(DataValidationError):
        predict(pd.DataFrame([{"mean_radius": 10.0}]), model=model)
    with pytest.raises(DataValidationError):
        predict(pd.DataFrame(), model=model)


def test_predict_handles_missing_values_via_imputer(artifacts):
    model = load_model(artifacts["cfg"]["paths"]["model_path"])
    row = pd.read_csv(Path(artifacts["cfg"]["data"]["processed_dir"]) / "test.csv").head(1)
    row.loc[:, "mean_radius"] = np.nan
    assert len(predict(row, model=model, class_names={0: "m", 1: "b"})) == 1


def test_model_loads_and_predicts_in_a_separate_python_process(artifacts):
    cfg = artifacts["cfg"]
    test_csv = Path(cfg["data"]["processed_dir"]) / "test.csv"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MODEL_PATH": cfg["paths"]["model_path"]}
    res = subprocess.run(
        [sys.executable, "-m", "src.predict", "--csv", str(test_csv)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    lines = [json.loads(l) for l in res.stdout.strip().splitlines()]
    assert len(lines) >= 5
    # same numbers as in-process prediction
    model = joblib.load(cfg["paths"]["model_path"])
    local = predict(pd.read_csv(test_csv), model=model, class_names=cfg["data"]["class_names"])
    assert [l["prediction"] for l in lines] == [r["prediction"] for r in local]
