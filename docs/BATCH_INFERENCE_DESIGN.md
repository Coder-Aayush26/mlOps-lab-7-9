# B6. Extending the service to batch (CSV) inference

**Implemented now:** `POST /predict/batch` accepts up to 1000 validated JSON records synchronously, and `python -m src.predict --csv in.csv --out out.csv` scores a file offline with the same pipeline.

**Proposed design for "upload a CSV of many records":**

| Volume of CSV | Approach | Reason |
|---|---|---|
| up to a few thousand rows (< ~1-2 s of work) | **Synchronous endpoint** `POST /predict/csv` (multipart upload, validate columns, return CSV/JSON) | simplest; client waits, no extra infrastructure |
| tens of thousands of rows | **Background job**: `POST /jobs` stores the file and returns `202 + job_id`; a worker (FastAPI `BackgroundTasks` for a prototype, Celery/RQ/Cloud queue for production) scores in chunks; `GET /jobs/{id}` returns status; `GET /jobs/{id}/result` downloads the output | avoids HTTP time-outs and keeps API latency low for online users |
| millions of rows / nightly scoring | **Separate batch process** (scheduled container, cron/Airflow, Spark) reading from and writing to a store (S3/DB) using the same `predict()` function | throughput matters more than latency; scale independently of the online service |

Shared requirements for every option: validate the schema row-by-row and report bad rows instead of failing the whole file; cap file size; score in chunks (e.g. 10k rows) to bound memory; never load the model per request (load once); return the model version with every result; log counts/timings; do not log record contents.

Recommendation for this project: keep `/predict` for online use, keep `/predict/batch` for small payloads, and run large files with the CLI/scheduled container so a heavy job can never starve the online API.
