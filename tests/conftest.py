"""Shared fixtures. Tests train a small model into a temp dir; they never touch models/ or data/."""
import copy
from pathlib import Path

import pytest
import yaml

from src.data import export_breast_cancer_csv
from src.train import run_training

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def base_cfg():
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory, base_cfg):
    """Train once per test session; return paths + config used."""
    tmp = tmp_path_factory.mktemp("artifacts")
    cfg = copy.deepcopy(base_cfg)
    raw = export_breast_cancer_csv(tmp / "raw.csv", cfg["data"]["features"])
    cfg["data"]["raw_path"] = str(raw)
    cfg["data"]["processed_dir"] = str(tmp / "processed")
    cfg["paths"] = {
        "model_path": str(tmp / "models" / "model.joblib"),
        "metadata_path": str(tmp / "models" / "model_metadata.json"),
        "reports_dir": str(tmp / "reports"),
    }
    cfg["model"]["n_estimators"] = 30
    cfg["mlflow"]["enabled"] = False
    metrics = run_training(cfg)
    return {"cfg": cfg, "metrics": metrics, "tmp": tmp, "raw": raw}


@pytest.fixture()
def valid_payload():
    return {
        "mean_radius": 14.1, "mean_texture": 19.3, "mean_perimeter": 92.0, "mean_area": 654.9,
        "mean_smoothness": 0.096, "mean_compactness": 0.104, "mean_concavity": 0.089,
        "mean_concave_points": 0.049, "mean_symmetry": 0.181, "mean_fractal_dimension": 0.063,
    }
