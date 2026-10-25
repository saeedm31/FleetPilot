# One Dockerfile, three images:
#   app     Streamlit app + copilot (non-root, health check)
#   train   batch job: simulate, train, score, register the model in MLflow
#   mlflow  MLflow tracking server + model registry + trace UI
# Run all of it: docker compose up   (see README)

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLEETPILOT_DATA=/data
WORKDIR /app
RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app /data
COPY requirements.txt requirements-mlflow.txt ./
RUN pip install -r requirements.txt -r requirements-mlflow.txt
COPY pyproject.toml ./
COPY fleetpilot/ fleetpilot/

FROM base AS train
RUN pip install "pytest>=8.0"
COPY tests/ tests/
USER app
CMD ["python", "-m", "fleetpilot", "build"]

FROM base AS app
COPY .streamlit/ .streamlit/
COPY demo/ demo/
COPY app.py ./
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

FROM python:3.12-slim AS mlflow
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install "mlflow>=3.1,<3.2" "sqlalchemy<2.1" && useradd --create-home --uid 1000 mlflow && mkdir -p /mlflow && chown mlflow /mlflow
USER mlflow
EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts"]
