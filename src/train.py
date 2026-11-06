"""Training stage: read config, prepare data, train the pipeline, save, evaluate, log to MLflow.

Run:  python -m src.train --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import RatioFeatures, load_data, preprocess_data, split_data
from src.evaluate import evaluate_model

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def build_estimator(model_cfg: dict[str, Any], random_state: int):
    kind = model_cfg["type"]
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_cfg["n_estimators"], max_depth=model_cfg["max_depth"],
            random_state=random_state, n_jobs=1,
        )
    if kind == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=model_cfg["n_estimators"], max_depth=model_cfg["max_depth"],
            random_state=random_state,
        )
    if kind == "logistic_regression":
        return LogisticRegression(C=model_cfg.get("C", 1.0), max_iter=model_cfg.get("max_iter", 1000),
                                  random_state=random_state)
    raise ValueError(f"Unknown model type: {kind!r}")


def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    """Preprocessing + estimator in ONE object so training and inference cannot diverge."""
    return Pipeline(
        [
            ("features", RatioFeatures()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", build_estimator(cfg["model"], cfg["training"]["random_state"])),
        ]
    )


def train_model(X_train, y_train, cfg: dict[str, Any]) -> Pipeline:
    """Fit the full pipeline on TRAINING data only."""
    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)
    return pipe


def prepare_datasets(cfg: dict[str, Any]):
    """load -> clean -> split; saves train/test CSVs for the evaluate stage."""
    d, t = cfg["data"], cfg["training"]
    raw = load_data(d["raw_path"])
    clean = preprocess_data(raw, d["features"], d["target"], d.get("strictly_positive", []))
    X_train, X_test, y_train, y_test = split_data(
        clean, d["features"], d["target"], t["test_size"], t["random_state"]
    )
    out = Path(d["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{d["target"]: y_train}).to_csv(out / "train.csv", index=False)
    X_test.assign(**{d["target"]: y_test}).to_csv(out / "test.csv", index=False)
    return X_train, X_test, y_train, y_test


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _log_to_mlflow(cfg, metrics, pipe, X_train) -> str | None:
    """Best-effort MLflow logging. Never breaks training if MLflow is unavailable."""
    if not cfg.get("mlflow", {}).get("enabled", False):
        return None
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("mlflow not installed; skipping experiment tracking.")
        return None
    try:
        uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name=f"{cfg['model']['type']}") as run:
            mlflow.log_params({f"model.{k}": v for k, v in cfg["model"].items()})
            mlflow.log_params({f"training.{k}": v for k, v in cfg["training"].items()})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_artifacts(cfg["paths"]["reports_dir"], artifact_path="reports")
            # cloudpickle: the pipeline contains our custom RatioFeatures transformer (skops would reject it)
            kwargs = dict(input_example=X_train.head(3), serialization_format="cloudpickle")
            try:
                mlflow.sklearn.log_model(pipe, name="model", **kwargs)
            except TypeError:  # MLflow 2.x signature
                mlflow.sklearn.log_model(pipe, artifact_path="model", **kwargs)
            logger.info("Logged run %s to MLflow at %s", run.info.run_id, uri)
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001 - tracking must never kill training
        logger.warning("MLflow logging failed (%s); training result is unaffected.", exc)
        return None


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    """Full training run. Returns metrics (and writes model, metadata, reports)."""
    t = cfg["training"]
    X_train, X_test, y_train, y_test = prepare_datasets(cfg)
    logger.info("Train rows: %d | Test rows: %d", len(X_train), len(X_test))

    pipe = train_model(X_train, y_train, cfg)

    cv = cross_val_score(build_pipeline(cfg), X_train, y_train, cv=t["cv_folds"], scoring="f1")
    logger.info("CV F1 on training data: %.4f +/- %.4f", cv.mean(), cv.std())

    model_path = Path(cfg["paths"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    logger.info("Saved inference pipeline to %s", model_path)

    metrics = evaluate_model(
        model_path, Path(cfg["data"]["processed_dir"]) / "test.csv", cfg["data"]["target"],
        cfg["paths"]["reports_dir"], cfg["data"].get("class_names"),
        extra={"cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std())},
    )
    run_id = _log_to_mlflow(cfg, metrics, pipe, X_train)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": cfg["model"]["type"],
        "model_params": {k: v for k, v in cfg["model"].items() if k != "type"},
        "features": cfg["data"]["features"],
        "engineered_features": ["area_to_perimeter", "radius_to_texture"],
        "target": cfg["data"]["target"],
        "class_names": {str(k): v for k, v in cfg["data"]["class_names"].items()},
        "random_state": t["random_state"],
        "test_size": t["test_size"],
        "dataset_sha256": _sha256(cfg["data"]["raw_path"]),
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, float)},
        "mlflow_run_id": run_id,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    Path(cfg["paths"]["metadata_path"]).write_text(json.dumps(metadata, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and save the inference pipeline.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging for this run")
    parser.add_argument("--model-type", help="Override model.type from the config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.no_mlflow:
        cfg["mlflow"]["enabled"] = False
    if args.model_type:
        cfg["model"]["type"] = args.model_type
    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
