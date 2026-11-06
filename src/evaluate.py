"""Evaluation stage: load model + held-out data, compute metrics, write reports."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.predict import feature_names, load_model  # noqa: E402

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str | Path,
    test_path: str | Path,
    target: str,
    report_dir: str | Path,
    class_names: Mapping[Any, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a saved pipeline on held-out data and save reports.

    Writes metrics.json, classification_report.txt, evaluation_report.md,
    confusion_matrix.png and roc_curve.png to ``report_dir``.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path)
    test = pd.read_csv(test_path)
    X, y = test[feature_names(model)], test[target]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, list(model.classes_).index(1)]
    names = {int(k): v for k, v in (class_names or {}).items()}
    labels = [names.get(int(c), str(c)) for c in model.classes_]

    metrics = {
        "n_test_samples": int(len(test)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
    }
    cm = confusion_matrix(y, y_pred, labels=list(model.classes_))
    metrics["confusion_matrix"] = cm.tolist()
    if extra:
        metrics.update(extra)

    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (report_dir / "classification_report.txt").write_text(
        classification_report(y, y_pred, target_names=labels, digits=4)
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (held-out test set)")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y, y_proba, ax=ax, name="pipeline")
    ax.set_title("ROC curve (held-out test set)")
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    lines = ["# Evaluation report", "", f"Held-out samples: **{metrics['n_test_samples']}**", "",
             "| Metric | Value |", "|---|---|"]
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        lines.append(f"| {k} | {metrics[k]:.4f} |")
    lines += ["", f"Confusion matrix (rows = true, cols = predicted, order = {labels}):", "", "```",
              str(cm), "```", ""]
    (report_dir / "evaluation_report.md").write_text("\n".join(lines))
    logger.info("Evaluation: %s", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the saved model on the held-out test set.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    evaluate_model(
        cfg["paths"]["model_path"],
        Path(cfg["data"]["processed_dir"]) / "test.csv",
        cfg["data"]["target"],
        cfg["paths"]["reports_dir"],
        cfg["data"].get("class_names"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
