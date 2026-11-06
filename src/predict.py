"""Prediction stage: load the saved inference pipeline and score new records."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

from src.data import DataValidationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/model.joblib"


def get_model_path(path: str | Path | None = None) -> Path:
    """Resolve model path: explicit arg > MODEL_PATH env var > default."""
    return Path(path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_model(path: str | Path | None = None):
    """Load the saved sklearn Pipeline (preprocessing + estimator)."""
    p = get_model_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)


def load_metadata(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load model_metadata.json that sits next to the model (empty dict if absent)."""
    meta = get_model_path(model_path).with_name("model_metadata.json")
    if meta.exists():
        return json.loads(meta.read_text())
    return {}


def feature_names(model) -> list[str]:
    return list(model.named_steps["features"].columns_)


def predict(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    model=None,
    class_names: Mapping[Any, str] | None = None,
) -> list[dict[str, Any]]:
    """Score records (list of dicts or a DataFrame). Returns one dict per record."""
    model = model if model is not None else load_model()
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    if df.empty:
        raise DataValidationError("No records supplied.")
    missing = [c for c in feature_names(model) if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required features: {missing}")
    df = df[feature_names(model)].apply(pd.to_numeric, errors="coerce")

    names = {int(k): v for k, v in (class_names or load_metadata().get("class_names", {})).items()}
    proba = model.predict_proba(df)
    classes = [int(c) for c in model.classes_]
    out = []
    for row in proba:
        idx = int(row.argmax())
        cls = classes[idx]
        out.append(
            {
                "prediction": cls,
                "label": names.get(cls, str(cls)),
                "probability": float(row[idx]),
                "probabilities": {names.get(c, str(c)): float(p) for c, p in zip(classes, row)},
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score records with the saved pipeline.")
    parser.add_argument("--model", default=None, help="Path to model.joblib")
    parser.add_argument("--csv", help="CSV of records to score (feature columns required)")
    parser.add_argument("--out", help="Where to write predictions CSV (with --csv)")
    parser.add_argument("--sample", type=int, default=0, help="Score first N rows of data/processed/test.csv")
    parser.add_argument("--json", help='One JSON record, e.g. \'{"mean_radius": 14.1, ...}\'')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model = load_model(args.model)
    meta = load_metadata(args.model)
    if args.json:
        records = pd.DataFrame([json.loads(args.json)])
    elif args.csv:
        records = pd.read_csv(args.csv)
    elif args.sample:
        records = pd.read_csv("data/processed/test.csv").head(args.sample)
    else:
        parser.error("provide --csv, --json or --sample N")

    results = predict(records, model=model, class_names=meta.get("class_names"))
    if args.csv and args.out:
        pd.concat([records.reset_index(drop=True), pd.DataFrame(results)[["prediction", "label", "probability"]]],
                  axis=1).to_csv(args.out, index=False)
        print(f"Wrote {len(results)} predictions to {args.out}")
    else:
        for i, r in enumerate(results):
            print(json.dumps({"row": i, **r}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
