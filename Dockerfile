# Base image used by all Kubeflow pipeline components
# Build: docker build -t fraud-detection:latest -f Dockerfile.training .
# Inside Minikube: eval $(minikube docker-env) first

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY pipeline/       ./pipeline/
COPY api/            ./api/
COPY scripts/        ./scripts/
COPY drift_simulation/ ./drift_simulation/
COPY explainability/   ./explainability/

# Create runtime directories
RUN mkdir -p /data /artifacts /serving /models /outputs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default entrypoint (overridden per component in KFP)
CMD ["python", "-m", "pipeline.pipeline"]
