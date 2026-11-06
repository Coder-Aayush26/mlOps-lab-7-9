# MLOps Lab 7-9: ML Pipeline -> FastAPI -> Docker

A reproducible breast-cancer classifier (benign / malignant) served over HTTP and packaged in Docker.
Data: scikit-learn's bundled public *Breast Cancer Wisconsin* dataset (10 "mean" features) - no personal or confidential data.

```
Data -> Cleaning -> Feature Engineering -> Training -> Evaluation -> Saved Pipeline (models/model.joblib)
                                                                        |
                                                              FastAPI  POST /predict
                                                                        |
                                                                 Docker container
```

## Repository layout
```
mlops-lab-7-9/
├── configs/config.yaml        # all changeable values (paths, seed, test size, hyper-parameters, MLflow)
├── data/raw/                  # breast_cancer.csv (tracked by DVC: breast_cancer.csv.dvc)
├── data/processed/            # train.csv / test.csv (generated)
├── models/                    # model.joblib (full sklearn Pipeline) + model_metadata.json
├── reports/                   # metrics.json, evaluation_report.md, confusion_matrix.png, roc_curve.png, ...
├── src/  data.py train.py evaluate.py predict.py api.py
├── tests/                     # 28 tests (data, pipeline, API)
├── scripts/                   # make_dataset.py, api_demo.sh, compare_predictions.py, docker_validate.sh, compose_validate.sh
├── docs/                      # CONCEPTS.md (A1/B1/C1), BATCH_INFERENCE_DESIGN.md (B6), VALIDATION_CHECKLIST.md, CHALLENGES.md
├── evidence/                  # captured outputs (local, api) + place for docker evidence / screenshots
├── Dockerfile  .dockerignore  docker-compose.yml
├── requirements.txt (runtime/API)  requirements-train.txt (+MLflow)  requirements-dev.txt (+pytest, httpx, DVC)
└── Report.pdf
```

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```
Python 3.11+ (tested on 3.12 locally; the Docker image uses 3.11). Versions are pinned because a pickled scikit-learn model must be loaded by the same scikit-learn version.

## 2. Data (DVC)
```bash
python scripts/make_dataset.py          # creates data/raw/breast_cancer.csv from the public dataset
dvc pull                                # alternative once a DVC remote is configured
```
The dataset is versioned with DVC (`data/raw/breast_cancer.csv.dvc` is committed, the CSV is git-ignored). To use your own remote: `dvc remote add -d storage <path-or-url>` then `dvc push`. After changing data: `dvc add data/raw/breast_cancer.csv && git add data/raw/breast_cancer.csv.dvc && git commit`.

## 3. Train, evaluate, predict (Part A)
```bash
python -m src.train                                  # clean -> split -> train -> save -> evaluate -> MLflow
python -m src.train --model-type logistic_regression # override config from the CLI
python -m src.train --no-mlflow                      # skip experiment tracking
python -m src.evaluate                               # re-evaluate saved model on data/processed/test.csv
python -m src.predict --sample 5                     # score first 5 held-out records (new Python process)
python -m src.predict --csv my.csv --out preds.csv   # batch score a CSV
python -m src.predict --json '{"mean_radius": 14.1, ...}'
```
Outputs: `models/model.joblib` (preprocessing + estimator in one sklearn Pipeline), `models/model_metadata.json`, `reports/*`.
Training reads `configs/config.yaml` (seed 42, test size 0.2, `random_forest` 100 trees, depth 5). MLflow tracking URI = `$MLFLOW_TRACKING_URI` or `sqlite:///mlflow.db`; browse runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Reference result (seed 42, 114 held-out rows): accuracy 0.9561, precision 0.9718, recall 0.9583, F1 0.9650, ROC-AUC 0.9917.

### Tests
```bash
python -m pytest -v            # 28 tests: cleaning, no leakage, separate-process load, API valid/invalid requests
```

## 4. API (Part B)
```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
Swagger UI: http://127.0.0.1:8000/docs - ReDoc: /redoc - OpenAPI: /openapi.json

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness: `{"status":"ok","model_loaded":true}` |
| GET | `/info` | model metadata, metrics, MLflow reachability |
| POST | `/predict` | online inference, one record |
| POST | `/predict/batch` | up to 1000 records (see `docs/BATCH_INFERENCE_DESIGN.md`) |

**Request** (`POST /predict`, all 10 fields required; sizes > 0, shape metrics in [0, 1], unknown fields rejected):
```json
{"mean_radius":14.1,"mean_texture":19.3,"mean_perimeter":92.0,"mean_area":654.9,
 "mean_smoothness":0.096,"mean_compactness":0.104,"mean_concavity":0.089,
 "mean_concave_points":0.049,"mean_symmetry":0.181,"mean_fractal_dimension":0.063}
```
**Response** (200):
```json
{"prediction":1,"label":"benign","probability":0.7783,"probabilities":{"malignant":0.2217,"benign":0.7783},
 "model_version":"2026-10-30T14:12:27+00:00"}
```
Errors: **422** (missing field / wrong type / out-of-range value, with per-field detail), **503** (model not loaded), **500** (`{"detail":"Internal server error."}` - details only in server logs).

### curl / Postman
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"mean_radius":14.1,"mean_texture":19.3,"mean_perimeter":92.0,"mean_area":654.9,"mean_smoothness":0.096,"mean_compactness":0.104,"mean_concavity":0.089,"mean_concave_points":0.049,"mean_symmetry":0.181,"mean_fractal_dimension":0.063}'
scripts/api_demo.sh            # valid, missing field, wrong type, invalid value, health, info
```
Postman: POST `http://127.0.0.1:8000/predict` -> Body -> raw -> JSON -> paste the request above -> Send.
Environment variables: `MODEL_PATH` (default `models/model.joblib`), `MLFLOW_TRACKING_URI` (optional), `LOG_LEVEL`.

## 5. Docker (Part C)
```bash
docker build -t ml-inference-api:1.0 .
docker image ls
docker run -d --name ml-api -p 8000:8000 ml-inference-api:1.0     # busy port? use -p 8001:8000
curl http://localhost:8000/health        # and open http://localhost:8000/docs
scripts/api_demo.sh http://localhost:8000
docker ps
docker logs ml-api
docker stop ml-api && docker rm ml-api   # clean up
```
Other options: mount a different model `-v "$PWD/models:/app/models:ro"`, change path with `-e MODEL_PATH=/app/models/model.joblib`.
The image: `python:3.11-slim`, runtime requirements only, non-root user, `HEALTHCHECK`, no secrets, `.dockerignore` excludes `.git`, `.venv`, data, notebooks, tests, reports.

One-shot evidence for C7 + Part D: `scripts/docker_validate.sh` (writes `evidence/docker/docker_validation.txt` and `reports/local_vs_container.md`).

### Docker Compose (API + MLflow)
```bash
docker compose config                    # validate
docker compose up --build -d             # api :8000, MLflow UI http://localhost:5000
docker compose ps && docker compose logs
curl http://localhost:8000/info          # "mlflow_reachable": true  => API -> http://mlflow:5000 works
docker compose --profile train run --rm trainer    # retrain in the stack and log to MLflow
docker compose down                      # keeps the mlflow-data volume (persistence)
docker compose down -v                   # also deletes it
```
`scripts/compose_validate.sh` automates these checks. The Compose file is a design scaffold: verify the MLflow tag/flags against the current MLflow docs before reusing it.

## 6. Final integration (Part D) - how a teammate reproduces everything
```bash
git clone <repo-url> && cd mlops-lab-7-9
python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
dvc pull || python scripts/make_dataset.py          # 1. data
python -m src.train                                  # 2-3. train, evaluate, save pipeline, log to MLflow
python -m pytest -q                                  #      tests
uvicorn src.api:app --port 8000                      # 4. API (second terminal): POST /predict
docker build -t ml-inference-api:1.0 .               # 5. image
docker run -d --name ml-api -p 8001:8000 ml-inference-api:1.0   # 6. container (port 8001 so it can run beside uvicorn)
python scripts/compare_predictions.py --url http://localhost:8001 --label container   # 7. local vs container
```
Consistency criterion: same class and |probability difference| <= 1e-6 (the container uses the same pinned libraries, so differences are expected to be 0; the tolerance allows for CPU/BLAS float differences).

## 7. Safety notes
Only public data is used. No credentials are stored in code, config or image; `.env` is git- and docker-ignored. Pickle/joblib files can execute code when loaded - only load model files you produced or trust.
