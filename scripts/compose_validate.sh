#!/usr/bin/env bash
# Verifies Docker Compose: config, API<->MLflow connectivity, persistence across `down`/`up`.
set -u
OUT=evidence/docker; mkdir -p "$OUT"
step() { echo; echo "##### $*"; }
{
step "docker compose config";      docker compose config --quiet && echo "compose file valid"
step "up --build";                 docker compose up --build -d; sleep 15
step "compose ps";                 docker compose ps
step "API -> MLflow connectivity"; curl -s http://localhost:8000/info | python -m json.tool   # expect mlflow_reachable: true
step "Retrain inside the stack (logs run to http://mlflow:5000)"; docker compose --profile train run --rm trainer
step "Runs visible in MLflow";     curl -s -X POST http://localhost:5000/api/2.0/mlflow/experiments/search -H 'Content-Type: application/json' -d '{"max_results":10}'
step "compose logs (tail)";        docker compose logs --tail=30
step "down (volume kept) then up"; docker compose down && docker compose up -d && sleep 15
step "Persistence check: experiments still present"; curl -s -X POST http://localhost:5000/api/2.0/mlflow/experiments/search -H 'Content-Type: application/json' -d '{"max_results":10}'
step "Cleanup (use 'docker compose down -v' to also delete the MLflow volume)"; docker compose down
} 2>&1 | tee "$OUT/compose_validation.txt"
