"""Compare predictions from the local pipeline (in-process) with a running API (local uvicorn or Docker).

Usage:
  python scripts/compare_predictions.py --url http://localhost:8000 --label container --n 10
Writes reports/local_vs_<label>.md and exits non-zero if any prediction differs beyond --tol.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.predict import load_metadata, load_model, predict  # noqa: E402


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # ignore HTTP(S)_PROXY for localhost
    with opener.open(req, timeout=10) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", default="container")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-6, help="max allowed |probability difference|")
    ap.add_argument("--test-csv", default="data/processed/test.csv")
    args = ap.parse_args()

    model, meta = load_model(), load_metadata()
    rows = pd.read_csv(args.test_csv).head(args.n)[meta["features"]]
    local = predict(rows, model=model, class_names=meta["class_names"])

    lines = [f"# Local vs {args.label} predictions", "", f"API: `{args.url}` | tolerance on probability: {args.tol:g}", "",
             "| # | local class | api class | local P | api P | abs diff | match |", "|---|---|---|---|---|---|---|"]
    ok = True
    for i, (rec, loc) in enumerate(zip(rows.to_dict("records"), local)):
        api = post(args.url.rstrip("/") + "/predict", rec)
        diff = abs(loc["probability"] - api["probability"])
        match = loc["prediction"] == api["prediction"] and diff <= args.tol
        ok &= match
        lines.append(f"| {i} | {loc['label']} | {api['label']} | {loc['probability']:.8f} | {api['probability']:.8f} | {diff:.2e} | {'yes' if match else 'NO'} |")
    lines += ["", f"**Result: {'ALL MATCH' if ok else 'MISMATCH'}**"]
    out = Path("reports") / f"local_vs_{args.label}.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
