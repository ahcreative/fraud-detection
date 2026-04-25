#!/bin/bash
# ============================================================
# kubeflow_setup.sh
# Sets up Kubeflow Pipelines on Minikube from scratch.
# Run in WSL Ubuntu on Windows 11.
# Assumes: minikube, kubectl, docker are already installed.
# ============================================================

set -e

MINIKUBE_MEMORY="8192"
MINIKUBE_CPUS="4"
MINIKUBE_DISK="30g"
KFP_VERSION="2.0.3"

echo ""
echo "========================================================"
echo "  Kubeflow Pipelines Setup on Minikube"
echo "  Memory: ${MINIKUBE_MEMORY}MB | CPUs: ${MINIKUBE_CPUS}"
echo "========================================================"

# ── Step 1: Start Minikube ────────────────────────────────────────────────────
echo ""
echo "[1/6] Starting Minikube ..."
minikube status &>/dev/null && echo "  Minikube already running" || \
minikube start \
    --memory="${MINIKUBE_MEMORY}" \
    --cpus="${MINIKUBE_CPUS}" \
    --disk-size="${MINIKUBE_DISK}" \
    --driver=docker \
    --kubernetes-version=v1.26.1

echo "  ✅ Minikube running"
kubectl cluster-info

# ── Step 2: Install Kubeflow Pipelines ───────────────────────────────────────
echo ""
echo "[2/6] Installing Kubeflow Pipelines v${KFP_VERSION} ..."
PIPELINE_VERSION="${KFP_VERSION}"

# Apply KFP standalone manifests
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}" || true
kubectl wait --for condition=established --timeout=60s \
    crd/applications.app.k8s.io || true

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=${PIPELINE_VERSION}" || true

echo "  Waiting for Kubeflow pods to be ready (this takes 5-10 minutes) ..."
kubectl wait --for=condition=Ready pods --all -n kubeflow --timeout=600s || \
    echo "  Some pods still starting — check: kubectl get pods -n kubeflow"

echo "  ✅ Kubeflow Pipelines installed"

# ── Step 3: Apply fraud detection namespace ───────────────────────────────────
echo ""
echo "[3/6] Creating fraud-detection namespace ..."
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/persistent-volumes.yaml
echo "  ✅ Namespace and PVCs created"

# ── Step 4: Build Docker image inside Minikube ───────────────────────────────
echo ""
echo "[4/6] Building Docker image inside Minikube ..."
eval "$(minikube docker-env)"
docker build -t fraud-detection:latest -f Dockerfile .
echo "  ✅ Image built: fraud-detection:latest"

# ── Step 5: Copy data to Minikube ─────────────────────────────────────────────
echo ""
echo "[5/6] Setting up data directory in Minikube ..."
minikube ssh "sudo mkdir -p /data"

if [ -f "./data/train_transaction.csv" ]; then
    echo "  Copying train_transaction.csv (this may take a few minutes) ..."
    minikube cp ./data/train_transaction.csv /data/train_transaction.csv
    minikube cp ./data/train_identity.csv    /data/train_identity.csv
    echo "  ✅ Data files copied"
else
    echo "  ⚠️  Data files not found in ./data/"
    echo "     Copy manually: minikube cp ./data/train_transaction.csv /data/"
fi

# ── Step 6: Start port-forward ────────────────────────────────────────────────
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "  To access Kubeflow UI, run in a new terminal:"
echo "    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80"
echo "  Then open: http://localhost:8888"
echo ""
echo "  To submit the pipeline:"
echo "    source venv/bin/activate"
echo "    python scripts/submit_kubeflow.py --endpoint http://localhost:8888"
echo ""
echo "  To check pod status:"
echo "    kubectl get pods -n kubeflow"
echo ""
