"""Build Report.pdf from the repository's real artefacts (metrics, evidence, config)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / "models/model_metadata.json").read_text())
metrics = json.loads((ROOT / "reports/metrics.json").read_text())
api_txt = (ROOT / "evidence/api/curl_results.txt").read_text()
cmp_md = (ROOT / "evidence/api/local_vs_api_local.md").read_text()

# ---------- architecture figure ----------
def architecture(path):
    fig, ax = plt.subplots(figsize=(10, 3.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 36)
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec="#333", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    steps = ["Data\nload_data", "Cleaning\npreprocess", "Features\nRatio+Impute\n+Scale", "Training\nRandomForest", "Evaluation\nmetrics/plots", "Saved\nPipeline\n.joblib"]
    for i, s in enumerate(steps):
        box(1 + i * 16.3, 22, 13, 10, s, "#dbeafe" if i < 5 else "#fde68a")
        if i < 5: ax.annotate("", xy=(1 + (i + 1) * 16.3 - 0.5, 27), xytext=(1 + i * 16.3 + 13.5, 27), arrowprops=dict(arrowstyle="->"))
    box(68, 6, 14, 9, "FastAPI\nPOST /predict\nGET /health", "#bbf7d0")
    box(86, 6, 12, 9, "Docker\ncontainer\n:8000", "#fecaca")
    box(40, 6, 22, 9, "MLflow tracking\n(params, metrics,\nmodel)", "#e9d5ff")
    ax.annotate("", xy=(75, 15.5), xytext=(91, 22), arrowprops=dict(arrowstyle="->", color="#999"))
    ax.annotate("", xy=(75, 15.8), xytext=(91.5, 21.5), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(51, 21.5), xytext=(56, 15.5), arrowprops=dict(arrowstyle="<-"))
    ax.annotate("", xy=(85.5, 10.5), xytext=(82.5, 10.5), arrowprops=dict(arrowstyle="<->"))
    ax.annotate("", xy=(62.5, 10.5), xytext=(67.5, 10.5), arrowprops=dict(arrowstyle="<->", ls="--"))
    ax.text(50, 1.5, "Same Pipeline object is used by training, evaluation, CLI prediction and the API (no training-serving skew)", ha="center", fontsize=8, style="italic")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)

arch = ROOT / "reports/architecture.png"; architecture(arch)

# ---------- styles ----------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, textColor=colors.HexColor("#1e3a8a"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, textColor=colors.HexColor("#1e40af"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=7, leading=8.5, backColor=colors.HexColor("#f3f4f6"), leftIndent=4)
def P(t, s=B): return Paragraph(t, s)
def code(t, n=110):
    lines = []
    for ln in t.strip("\n").splitlines():
        while len(ln) > n: lines.append(ln[:n]); ln = "    " + ln[n:]
        lines.append(ln)
    return Preformatted("\n".join(lines), CODE)
def table(rows, widths, head=True, fs=8):
    data = [[Paragraph(str(c), ParagraphStyle("c", parent=B, fontSize=fs, leading=fs + 2)) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")))
    t.setStyle(TableStyle(st)); return t
def img(path, w):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    return Image(str(path), width=w * cm, height=w * cm * ih / iw)
PENDING = "<font color='#b45309'><b>RUN ON YOUR MACHINE</b></font>"
OK = "<font color='#047857'><b>VERIFIED</b></font>"

S = []
S += [P("MLOps Lab Assignment: ML Pipelines, FastAPI and Docker", ParagraphStyle("T", parent=ss["Title"], fontSize=20)),
      P("Sessions 7, 8 and 9 - Report", ParagraphStyle("st", parent=B, alignment=1, fontSize=12)),
      P("Course: MLOps: From Machine Learning to Production AI Systems", ParagraphStyle("st2", parent=B, alignment=1)),
      Spacer(1, 6),
      P("<b>Evidence key.</b> " + OK + " = executed when this project was generated (outputs stored under <i>evidence/</i> and <i>reports/</i>). "
        + PENDING + " = requires Docker or a browser/Postman, which were not available in the generation environment; the exact command that produces the evidence is given, and "
        "<i>evidence/docker/</i> and <i>evidence/api/screenshots/</i> are the places to store it before submission."), Spacer(1, 4)]

# 1 Objectives
S += [P("1. Objectives", H1),
      P("Turn a trained classification model into a reproducible service: (1) refactor notebook-style code into a modular pipeline (data, train, evaluate, predict) driven by a configuration file; "
        "(2) expose predictions through a validated FastAPI REST API (<i>POST /predict</i>, <i>GET /health</i>); (3) package the service with Docker (and explore Docker Compose with an MLflow Tracking Server); "
        "(4) document everything so another developer can reproduce it, and show that local and containerised predictions agree within a documented tolerance."),
      P("<b>Problem and data.</b> Binary classification (benign vs malignant) on scikit-learn's bundled public Breast Cancer Wisconsin dataset: 569 rows, 10 numeric 'mean' features "
        "(radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension). Public data only; no personal or confidential information. "
        "The CSV is versioned with DVC (<i>data/raw/breast_cancer.csv.dvc</i>).")]

# 2 Architecture
S += [P("2. Architecture", H1), img(arch, 17),
      P("The inference artefact is a single scikit-learn <i>Pipeline</i> (<i>RatioFeatures</i> -> median <i>SimpleImputer</i> -> <i>StandardScaler</i> -> <i>RandomForestClassifier</i>), so identical preprocessing is applied "
        "in training, evaluation, CLI prediction and the API."),
      table([["Module", "Responsibility", "Key functions"],
             ["src/data.py", "Load, validate, clean, feature engineering transformer, split", "load_data, validate_schema, preprocess_data, RatioFeatures, split_data"],
             ["src/train.py", "Read config, train, save pipeline + metadata, call evaluation, log to MLflow", "load_config, build_pipeline, train_model, run_training, main"],
             ["src/evaluate.py", "Load model + held-out data, metrics, plots, reports", "evaluate_model, main"],
             ["src/predict.py", "Load saved pipeline, score records, CLI", "load_model, load_metadata, predict, main"],
             ["src/api.py", "FastAPI service, schemas, error handling", "/health, /info, /predict, /predict/batch"]], [3, 8, 6])]

# 3 Part A
S += [PageBreak(), P("3. Part A - ML pipeline (Session 7)", H1), P("3.1 Pipeline stages (A1)", H2),
      table([["Stage", "Input", "Output", "Purpose", "Possible failure"],
             ["Data loading", "CSV path", "raw DataFrame", "Bring approved data into memory", "Missing file / wrong columns"],
             ["Cleaning", "raw DataFrame", "clean DataFrame", "Drop duplicates, fix formats, flag invalid values", "Treating valid zeros as invalid; dropping too many rows"],
             ["Feature engineering", "clean features", "model-ready matrix", "Ratios, imputation, scaling", "Fitting scaler on all data (leakage); column-order skew"],
             ["Training", "X_train, y_train, config", "fitted Pipeline", "Learn parameters", "Bad hyper-parameters; non-reproducible seed"],
             ["Evaluation", "saved model, held-out set", "metrics + reports", "Measure generalisation", "Evaluating on training data"],
             ["Prediction", "saved model, new records", "label + probability", "Score unseen data", "Renamed/missing columns; stale model"]],
            [2.4, 2.8, 2.8, 4.6, 4.4], fs=7.5),
      P("Each stage has one responsibility so it can be tested, replaced and reasoned about independently (see docs/CONCEPTS.md)."),
      P("3.2 Configuration and execution (A3)", H2),
      P("All changeable values live in <i>configs/config.yaml</i> (dataset path, target, feature list, seed 42, test size 0.2, CV folds, model type and hyper-parameters, output paths, MLflow settings). "
        "CLI flags (<i>--config</i>, <i>--model-type</i>, <i>--no-mlflow</i>) and the <i>MLFLOW_TRACKING_URI</i> environment variable override it."),
      code("python -m src.train --config configs/config.yaml\npython -m src.evaluate\npython -m src.predict --sample 5"),
      P("3.3 Results", H2),
      table([["Metric (114 held-out rows)", "Value"], ["Accuracy", f"{metrics['accuracy']:.4f}"], ["Precision", f"{metrics['precision']:.4f}"],
             ["Recall", f"{metrics['recall']:.4f}"], ["F1", f"{metrics['f1']:.4f}"], ["ROC-AUC", f"{metrics['roc_auc']:.4f}"],
             ["5-fold CV F1 (training data only)", f"{meta['metrics']['cv_f1_mean']:.4f} +/- {meta['metrics']['cv_f1_std']:.4f}"],
             ["Confusion matrix [[TN,FP],[FN,TP]] (malignant, benign)", str(metrics['confusion_matrix']).replace("],", "], ")]], [9, 6]),
      Spacer(1, 4),
      Table([[img(ROOT / "reports/confusion_matrix.png", 7.5), img(ROOT / "reports/roc_curve.png", 7.5)]]),
      P("Figure: confusion matrix and ROC curve on the held-out test set (reports/). The model's recall for the malignant class is 0.952 (40/42 detected); in a real clinical setting "
        "this metric and the cost of false negatives would need much closer scrutiny - this lab model is a teaching artefact, not a diagnostic tool.", SM),
      P("3.4 Validation (A4) " + OK, H2),
      table([["Check", "Evidence", "Result"],
             ["Run from terminal", "evidence/local/train_log.txt", "Completed; model saved, 455 train / 114 test rows"],
             ["Import has no side effects", "test_importing_modules_does_not_start_training", "Pass - nothing written on import"],
             ["No leakage into preprocessing", "test_test_data_not_used_to_fit_preprocessing", "Pass - scaler mean == train mean (!= full-data mean); train/test disjoint"],
             ["Load in separate process", "test_model_loads_and_predicts_in_a_separate_python_process", "Pass - predictions identical to in-process"],
             ["5 predictions", "evidence/local/predict_5_records.txt", "5 valid outputs (e.g. malignant 0.9996, benign 0.9965)"],
             ["Tests", "evidence/local/pytest_verbose.txt", "28 passed (data 6, pipeline 8, API 14 incl. parametrised cases)"]],
            [4, 6.5, 6.5]),
      P("3.5 Experiment tracking", H2),
      P(f"Runs are logged to MLflow (params, metrics, report artefacts and the pipeline as a model). Verified run id: <i>{meta['mlflow_run_id']}</i> (experiment <i>mlops-lab-7-9</i>, status FINISHED, SQLite backend). "
        "Logging is best-effort: a tracking failure never aborts training. The pipeline is logged with cloudpickle because MLflow 3.x's default <i>skops</i> format rejected the custom transformer.")]

# 4 Part B
S += [PageBreak(), P("4. Part B - FastAPI serving (Session 8)", H1), P("4.1 REST and inference concepts (B1)", H2),
      P("REST exposes operations as HTTP requests/responses; GET reads (health), POST submits data for processing (prediction). The request schema defines required fields and types; the response schema defines the prediction and metadata. "
        "<b>Online inference</b> answers one or a few records immediately (latency/availability matter); <b>batch inference</b> scores a collection or file as a scheduled/asynchronous job (throughput/scheduling matter)."),
      P("4.2 Application (B2)", H2),
      table([["Endpoint", "Description"],
             ["GET /health", "{\"status\":\"ok\",\"model_loaded\":true}"],
             ["POST /predict", "10 required float fields (sizes > 0, shape metrics in [0,1], unknown fields rejected). Returns prediction, label, probability, per-class probabilities, model_version"],
             ["POST /predict/batch", "1-1000 records, same validation (optional extra)"],
             ["GET /info", "Model metadata and MLflow reachability (used to prove API -> MLflow connectivity in Compose)"]], [4, 13]),
      P("The model is loaded once at start-up (lifespan handler) from <i>MODEL_PATH</i>; the API builds a DataFrame with the exact training column names/order, so the saved Pipeline applies the same preprocessing. "
        "A global exception handler returns a generic 500 and logs details server-side only; a missing model gives 503 while <i>/health</i> stays alive."),
      P("4.3 Swagger UI and Postman/curl (B3, B4)", H2),
      P("Swagger UI is served at <i>/docs</i> (generated from the OpenAPI schema, saved as evidence/api/openapi.json). The curl calls below were executed against a live uvicorn process. "
        "Swagger UI/Postman screenshots: " + PENDING + " (open http://127.0.0.1:8000/docs, try GET /health and POST /predict, save screenshots in evidence/api/screenshots/)."),
      P("4.4 Validation and error handling (B5) " + OK, H2),
      table([["Test", "Expected", "Observed (live run)"],
             ["Valid input", "Prediction", "HTTP 200: prediction 1 (benign), probability 0.7783"],
             ["Missing field", "422", "HTTP 422, 'Field required' for each of the 8 missing fields"],
             ["Wrong type (\"large\")", "422", "HTTP 422 float_parsing on mean_radius"],
             ["Invalid business value", "Controlled error", "HTTP 422: radius -5 (must be > 0); smoothness 1.7 (must be <= 1)"],
             ["Health check", "Healthy", "HTTP 200 {\"status\":\"ok\",\"model_loaded\":true}"],
             ["Unexpected exception", "No leak", "HTTP 500 {\"detail\":\"Internal server error.\"}; service still healthy (test)"],
             ["Model file missing", "Controlled", "HTTP 503; /health still 200 (test)"]], [4, 3, 10]),
      P("Automated API tests (14, FastAPI TestClient) cover all rows above plus parametrised invalid values, unknown fields, batch limits and API-vs-local equality.", B),
      P("Example exchange (evidence/api/curl_results.txt):", SM),
      code(api_txt.split("=== 3.")[0].strip(), 120),
      P("4.5 Online versus batch design (B6)", H2),
      P("Implemented: small synchronous <i>/predict/batch</i> and an offline CSV CLI. Proposed for CSV upload: synchronous multipart endpoint for up to a few thousand rows; a background job (202 + job id, worker, status/result endpoints) for tens of thousands; "
        "and a separate scheduled batch container reading/writing a store for millions, so heavy scoring never starves the low-latency online API. Details and trade-offs: docs/BATCH_INFERENCE_DESIGN.md.")]

# 5 Part C
S += [PageBreak(), P("5. Part C - Docker (Session 9)", H1), P("5.1 Concepts (C1)", H2),
      P("<b>Image</b>: read-only template with code, dependencies and runtime configuration. <b>Container</b>: a running instance of an image. <b>Dockerfile</b>: build instructions (cached layers). "
        "<b>Port mapping</b>: <i>-p host:container</i>. <b>Volume</b>: storage outside the container's writable layer; a host path is not visible inside a container unless mounted. "
        "<b>Environment variable</b>: runtime configuration without rebuilding. <b>Compose</b>: several services, a shared network and volumes in one YAML file."),
      P("<b>Why code works on one laptop but not another:</b> different OS, Python/library versions (a pickled scikit-learn model depends on the scikit-learn version), missing system libraries, environment variables, relative paths, files present on only one machine, busy ports. "
        "Containers ship interpreter, libraries and files together, so the runtime environment is the same wherever the image runs."),
      P("<b>What containers do not guarantee:</b> identical hardware (ARM vs x86, GPU drivers), bit-identical floating-point results across CPUs, code security or correctness, reproducible rebuilds without pinned versions, data persistence without volumes, or resource limits unless configured."),
      P("5.2 Implementation (C2-C6)", H2),
      table([["Item", "Implementation"],
             ["Dockerfile", "python:3.11-slim; installs pinned requirements.txt first (layer cache); copies src/, models/, configs/; non-root user; HEALTHCHECK; CMD uvicorn on 0.0.0.0:8000; MODEL_PATH env var"],
             [".dockerignore", ".git, .venv, __pycache__, *.pyc, data/, reports/, tests/, notebooks, mlruns, .env (no secrets or large data in the image)"],
             ["Configuration", "Env vars MODEL_PATH, MLFLOW_TRACKING_URI, LOG_LEVEL; no credentials baked in"],
             ["Persistence", "Compose: named volume mlflow-data (SQLite DB + artefacts); ./models mounted read-only into the API so a retrained model needs no rebuild"],
             ["docker-compose.yml", "Services api (port 8000), mlflow (ghcr.io/mlflow/mlflow:v3.10.1, port 5000, SQLite backend, --serve-artifacts, --allowed-hosts for the service name) and an optional trainer (profile 'train') that logs to http://mlflow:5000"]],
            [3.5, 13.5]),
      code(open(ROOT / "Dockerfile").read(), 115),
      P("<b>Implemented vs proposed.</b> The Dockerfile, .dockerignore, Compose file and validation scripts are implemented and syntactically checked (Compose YAML parses). The Compose file is a single-host design scaffold; the MLflow image tag, flags (including <i>--allowed-hosts</i>, required because recent MLflow validates the Host header) "
        "and storage layout were taken from current MLflow documentation but have not been executed in the generation environment, which had no Docker daemon.", B),
      P("5.3 Container validation (C7)", H2),
      table([["Check", "Command", "Status"],
             ["14 Build image", "docker build -t ml-inference-api:1.0 .", PENDING],
             ["15 Run + /health", "docker run -d --name ml-api -p 8000:8000 ml-inference-api:1.0 ; curl localhost:8000/health", PENDING],
             ["16 Prediction from host", "scripts/api_demo.sh http://localhost:8000", PENDING],
             ["17 Logs", "docker logs ml-api", PENDING],
             ["18 Remove / recreate", "docker rm -f ml-api ; docker run ... ; curl /health", PENDING],
             ["19 Compose connectivity + persistence", "scripts/compose_validate.sh (GET /info shows mlflow_reachable: true; experiments survive down/up)", PENDING]], [4.2, 10, 2.8]),
      P("<b>All of C7 and the Part D comparison are produced by one command:</b> <i>scripts/docker_validate.sh</i> (writes evidence/docker/docker_validation.txt and reports/local_vs_container.md). "
        "Paste the build output, <i>docker image ls</i>, <i>docker logs</i> and screenshots into Section 5.3 of your final submission.")]

# 6 Part D
S += [P("6. Part D - Final integration", H1),
      table([["Step", "Command", "Status"],
             ["1 Prepare data + train (modular)", "python scripts/make_dataset.py ; python -m src.train", OK],
             ["2 Evaluate + save pipeline", "reports/*, models/model.joblib", OK],
             ["3 MLflow tracking", "run " + meta['mlflow_run_id'][:12] + "... FINISHED", OK],
             ["4 FastAPI + POST /predict", "uvicorn src.api:app ; scripts/api_demo.sh", OK],
             ["5 Build image", "docker build -t ml-inference-api:1.0 .", PENDING],
             ["6 Run container + predict from host", "docker run ... ; curl /predict", PENDING],
             ["7 Teammate reproduction", "README section 6", OK]], [5, 9, 3]),
      P("<b>Local vs service consistency.</b> Ten held-out records were scored in-process (<i>predict()</i>) and through the running API; classes and probabilities were identical (max |difference| = 0.0, tolerance 1e-6). "
        "This verifies the API layer adds no preprocessing drift. The same script run against the container (<i>--label container</i>) completes the requirement; "
        "tolerance is documented because the same pinned libraries should give identical results, but different CPU/BLAS builds could introduce ~1e-12 differences."),
      code(cmp_md.split("**Result")[0].strip(), 120), P("Result: ALL MATCH (10/10)", B)]

# 7 reproduce, challenges, reflection
S += [P("7. Reproducing the workflow", H1),
      code("git clone <repo-url> && cd mlops-lab-7-9\npython -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt\n"
           "dvc pull || python scripts/make_dataset.py\npython -m src.train && python -m pytest -q\nuvicorn src.api:app --port 8000\n"
           "docker build -t ml-inference-api:1.0 . && docker run -d --name ml-api -p 8001:8000 ml-inference-api:1.0\n"
           "python scripts/compare_predictions.py --url http://localhost:8001 --label container"),
      P("Versions are pinned (scikit-learn 1.8.0, pandas 3.0.2, FastAPI 0.162.0, ...) because a pickled model must be loaded by the same scikit-learn version; seeds are fixed (42); data is versioned by DVC (MD5 in the .dvc file) and its SHA-256 is stored in model_metadata.json.", B),
      P("8. Challenges and resolutions", H1),
      table([["Challenge", "Resolution"],
             ["Scaling on all data would leak test information", "Imputer + scaler + estimator in one Pipeline fitted on the train split; leakage test"],
             ["Zero valid for some features, invalid for others", "Config list <i>strictly_positive</i>"],
             ["Custom transformer must be importable when unpickling in API/Docker", "Defined in src/data.py (not __main__); run via python -m; image copies src/"],
             ["MLflow 3.x skops rejected the custom transformer", "cloudpickle serialisation; tracking wrapped so it cannot break training"],
             ["Importing modules must not train", "All work behind main(); subprocess test"],
             ["Local HTTP calls went through a proxy in the sandbox", "urllib opener with ProxyHandler({}) for localhost"],
             ["Errors must not leak internals", "Typed validation (422), 503, generic 500 + server-side logging"],
             ["MLflow validates Host header", "--allowed-hosts including the Compose service name"],
             ["No Docker daemon in the generation environment", "Scripted validation + explicit VERIFIED / RUN-ON-YOUR-MACHINE labelling"]], [7, 10]),
      P("9. Reflection", H1),
      P("Packaging preprocessing with the estimator removed the most common source of production bugs (training-serving skew), and writing the leakage and separate-process tests turned 'it works in my notebook' into checkable claims. "
        "Strict request schemas moved most input errors to the edge of the system as clear 422 responses, and keeping the 500 handler generic protects internals. "
        "Docker adds reproducibility of the environment, not of the model's correctness or of hardware-level numerics, which is why a numerical tolerance is documented. "
        "Limitations: the dataset is small and public, the model is not clinically validated, pickle files are only safe from trusted sources, the Compose design is single-host and unauthenticated, and no load/latency testing was done. "
        "Next steps: CI to run tests and build the image, a model registry with promotion, authentication and rate limiting, and an asynchronous batch worker.")]

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.drawString(2 * cm, 1.2 * cm, "MLOps Lab 7-9 Report"); c.drawRightString(19 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()

doc = SimpleDocTemplate(str(ROOT / "Report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="MLOps Lab 7-9 Report", author="")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("Report.pdf written")
