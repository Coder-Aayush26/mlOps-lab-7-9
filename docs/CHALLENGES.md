# Challenges and resolutions

| # | Challenge | Resolution |
|---|---|---|
| 1 | Notebook code mixed loading, cleaning, scaling and modelling; scaling on all data would leak test information | Split into `data/train/evaluate/predict`; imputer + scaler + estimator live in ONE `sklearn.Pipeline` fitted on the train split only; leakage test added |
| 2 | Zero is a legitimate value for some columns (concavity) but not for others (radius) | Config lists `strictly_positive` columns; only those treat `<= 0` as invalid |
| 3 | Invalid values: drop rows or fix? | Marked as NaN and median-imputed *inside* the Pipeline (learned from train only), so API/CLI get the same behaviour |
| 4 | Custom transformer must be importable when the pickle is loaded (API, Docker) | Transformer lives in `src/data.py` (never `__main__`); training is launched as `python -m src.train`; image copies `src/` |
| 5 | MLflow 3.x logs sklearn models with `skops` by default and rejected the custom transformer | Logged with `serialization_format="cloudpickle"`; MLflow logging wrapped so a tracking failure can never break training |
| 6 | Importing a module must not train | All work is behind `main()` / `if __name__ == "__main__"`; subprocess test proves it |
| 7 | `urllib` in scripts went through an HTTP proxy and could not reach `127.0.0.1` in the dev sandbox | Use an opener with `ProxyHandler({})` for local calls |
| 8 | Pydantic must reject bad input without crashing, and 500s must not leak internals | Typed schema with constraints (422), 503 when no model, global handler returns a generic 500 and logs the stack trace server-side |
| 9 | Model file must exist inside the container | `COPY models/ ./models/`; compose also mounts `./models` read-only so a retrained model needs no rebuild |
| 10 | MLflow >= 3.5 validates the `Host` header, so `http://mlflow:5000` can be rejected | Compose starts MLflow with `--allowed-hosts "mlflow:*,localhost:*,127.0.0.1:*"` |
| 11 | Docker was not available in the environment that generated this project | Container deliverables are fully scripted (`docker_validate.sh`, `compose_validate.sh`); the Report separates VERIFIED from RUN-ON-YOUR-MACHINE items |
