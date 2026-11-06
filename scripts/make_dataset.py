"""Create data/raw/breast_cancer.csv from the public scikit-learn dataset.

Usage: python scripts/make_dataset.py [--out data/raw/breast_cancer.csv]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.data import export_breast_cancer_csv  # noqa: E402

if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=cfg["data"]["raw_path"])
    args = ap.parse_args()
    p = export_breast_cancer_csv(args.out, cfg["data"]["features"])
    print(f"Wrote {p}")
