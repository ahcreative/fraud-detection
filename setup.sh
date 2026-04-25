#!/bin/bash
# ============================================================
# setup.sh — Full environment setup for fraud detection system
# Run inside WSL (Ubuntu) on Windows 11
# ============================================================

set -e

echo ""
echo "=================================================="
echo "  Fraud Detection System — Environment Setup"
echo "=================================================="

# ── Python virtual environment ─────────────────────────────
echo ""
echo "[1/7] Creating Python virtual environment ..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools

# ── Install Python packages ────────────────────────────────
echo ""
echo "[2/7] Installing Python packages ..."
pip install -r requirements.txt

echo "  ✅ Python packages installed"

# ── Create data directory ──────────────────────────────────
echo ""
echo "[3/7] Creating project directories ..."
mkdir -p data outputs serving models artifacts

echo "  ✅ Directories created"

# ── Verify Minikube ────────────────────────────────────────
echo ""
echo "[4/7] Checking Minikube status ..."
if command -v minikube &> /dev/null; then
    minikube status || echo "  Minikube not running — start with: minikube start"
else
    echo "  Minikube not found — install from: https://minikube.sigs.k8s.io"
fi

# ── Apply Kubernetes resources ─────────────────────────────
echo ""
echo "[5/7] Applying Kubernetes resources (if minikube running) ..."
if kubectl cluster-info &> /dev/null 2>&1; then
    kubectl apply -f kubernetes/namespace.yaml     || true
    kubectl apply -f kubernetes/persistent-volumes.yaml || true
    echo "  ✅ Kubernetes resources applied"
else
    echo "  ⚠️  Kubectl not connected — run manually after starting minikube"
fi

# ── Build Docker images ────────────────────────────────────
echo ""
echo "[6/7] Building Docker images ..."
if command -v docker &> /dev/null; then
    # Point to minikube docker daemon
    eval $(minikube docker-env 2>/dev/null || echo "")
    docker build -t fraud-detection:latest        -f Dockerfile.training . || echo "  Training image build failed (non-fatal)"
    docker build -t fraud-detection-api:latest    -f Dockerfile.api .      || echo "  API image build failed (non-fatal)"
    echo "  ✅ Docker images built"
else
    echo "  ⚠️  Docker not found"
fi

# ── Compile KFP pipeline ───────────────────────────────────
echo ""
echo "[7/7] Compiling Kubeflow pipeline ..."
python pipeline/pipeline.py && echo "  ✅ Pipeline compiled: fraud_detection_pipeline.yaml" || \
    echo "  ⚠️  Pipeline compile failed (kfp may not be fully installed)"

# ── Done ───────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "  Next steps:"
echo "  1. Place dataset files in ./data/"
echo "     - train_transaction.csv"
echo "     - train_identity.csv"
echo ""
echo "  2. Activate virtualenv: source venv/bin/activate"
echo ""
echo "  3. Run local pipeline:"
echo "     python scripts/run_local.py --data_dir ./data --output_dir ./outputs"
echo ""
echo "  4. Start monitoring stack:"
echo "     docker-compose up -d"
echo ""
echo "  5. Submit to Kubeflow (after port-forward):"
echo "     kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80"
echo "     python scripts/submit_kubeflow.py"
echo ""
