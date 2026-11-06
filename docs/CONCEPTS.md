# Self-learning notes (A1, B1, C1)

## A1. Pipeline stages

| Stage | Input | Output | Purpose | One possible failure | Why separate? |
|---|---|---|---|---|---|
| Data loading (`data.load_data`) | CSV path | raw DataFrame | Get approved data into memory | File missing, wrong columns, empty file | Source changes (file -> DB) touch only this function |
| Cleaning (`data.preprocess_data`) | raw DataFrame | clean DataFrame | Remove duplicates, bad formats, invalid values | Silently dropping too many rows or treating valid zeros as invalid | Cleaning rules can be unit-tested without a model |
| Feature engineering (`data.RatioFeatures` + imputer + scaler, inside the Pipeline) | clean features | model-ready matrix | Create ratios, impute, scale | Fitting the scaler on all data (test leakage); different columns/order at inference (training-serving skew) | Packaged inside the saved Pipeline so training and inference are identical |
| Training (`train.train_model`) | X_train, y_train, config | fitted Pipeline | Learn model parameters | Wrong hyper-parameters, non-reproducible seed, overfitting | Config-driven; swapping the estimator does not touch data code |
| Evaluation (`evaluate.evaluate_model`) | saved model + held-out test set | metrics + reports | Measure generalisation honestly | Evaluating on training data; metric that hides class imbalance | Can be re-run on any model/data without retraining |
| Prediction (`predict.predict`) | saved model + new records | labels + probabilities | Score unseen records | Missing/renamed columns, stale model file | Same code path is used by CLI, API and tests |

## B1. REST and inference

* **REST API** – operations exposed as HTTP requests/responses to resources (URLs).
* **Methods** – GET reads (idempotent, no body: `/health`), POST submits data to be processed (`/predict`), PUT replaces a resource, DELETE removes it.
* **Request schema** – `PredictionInput` (Pydantic): 10 float fields with constraints (sizes > 0, shape metrics in [0, 1], no extra fields).
* **Response schema** – `PredictionOutput`: class id, label, probability, per-class probabilities, model version.
* **Online inference** – one/few records, answered immediately; the concern is latency and availability.
* **Batch inference** – a file/collection scored as a (scheduled/async) job; the concern is throughput and scheduling. See `docs/BATCH_INFERENCE_DESIGN.md`.

## C1. Docker

* **Image** – read-only template: OS layer + Python + dependencies + code. **Container** – a running instance of an image with a thin writable layer.
* **Dockerfile** – recipe for building an image; each instruction is a cached layer (hence `COPY requirements.txt` + `pip install` *before* `COPY src/`).
* **Port mapping** – `-p 8000:8000` connects host port 8000 to container port 8000; the container's own network is otherwise isolated.
* **Volume** – storage outside the container's writable layer; survives `docker rm`. A host path is **not** visible in a container unless mounted.
* **Environment variable** – runtime configuration (`MODEL_PATH`, `MLFLOW_TRACKING_URI`) without rebuilding the image.
* **Docker Compose** – declares several services, a shared network (service-name DNS such as `http://mlflow:5000`) and volumes in one YAML file.

**Why "works on my laptop" fails elsewhere:** different OS, Python version, library versions (a pickled scikit-learn model can fail or silently change across versions), missing system libraries, different environment variables, relative paths, files that exist only on one machine, or different ports in use. A container ships the interpreter, libraries and files together, so the runtime environment is the same everywhere the image runs.

**What containers do not guarantee:** identical *hardware* (CPU architecture such as ARM vs x86, GPU drivers), bit-identical floating-point results across CPUs/BLAS builds (hence a numerical tolerance), correct or secure code (a vulnerable dependency is shipped as-is), reproducible rebuilds unless versions are pinned (base image tags and pip packages can change), data or model correctness, persistence of data without volumes, or performance/resource limits unless configured.
