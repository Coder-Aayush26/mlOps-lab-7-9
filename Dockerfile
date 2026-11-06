FROM python:3.11-slim

# REQUIREMENTS_FILE lets Compose build a "trainer" image with the extra training deps
ARG REQUIREMENTS_FILE=requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO

WORKDIR /app

COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

COPY src/ ./src/
COPY models/ ./models/
# configs are tiny and let the same image run `python -m src.train` (trainer service)
COPY configs/ ./configs/

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
