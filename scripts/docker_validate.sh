#!/usr/bin/env bash
# Runs checklist C7 (and the Part D comparison) and stores evidence in evidence/docker/.
# Usage: scripts/docker_validate.sh [HOST_PORT]    (default 8000)
set -u
PORT="${1:-8000}"
IMG=ml-inference-api:1.0
NAME=ml-api
OUT=evidence/docker
mkdir -p "$OUT"
step() { echo; echo "##### $*"; }

{
step "C7.14 Build the image";            docker build -t $IMG . 
step "Image details";                    docker image ls $IMG; docker image inspect $IMG --format 'Size={{.Size}} bytes, Created={{.Created}}'
step "C7.15 Run container";              docker rm -f $NAME >/dev/null 2>&1; docker run -d --name $NAME -p $PORT:8000 $IMG
sleep 6
step "docker ps";                        docker ps --filter name=$NAME
step "C7.15 /health from host";          curl -s -w "\nHTTP_STATUS=%{http_code}\n" http://localhost:$PORT/health
step "C7.16 valid prediction from host"; scripts/api_demo.sh http://localhost:$PORT
step "C7.17 container logs";             docker logs $NAME
step "Health status (HEALTHCHECK)";      docker inspect --format '{{.State.Health.Status}}' $NAME
step "Part D: local vs container";       python scripts/compare_predictions.py --url http://localhost:$PORT --label container --n 10
step "C7.18 remove + recreate container"; docker stop $NAME && docker rm $NAME && docker run -d --name $NAME -p $PORT:8000 $IMG && sleep 6 && curl -s http://localhost:$PORT/health
step "Cleanup";                          docker stop $NAME && docker rm $NAME
} 2>&1 | tee "$OUT/docker_validation.txt"
echo "Evidence saved to $OUT/docker_validation.txt and reports/local_vs_container.md"
