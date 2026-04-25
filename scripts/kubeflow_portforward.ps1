# kubeflow_portforward.ps1
# Run this in PowerShell to start Kubeflow UI port-forwarding
# Keep this window open while using the Kubeflow UI

Write-Host ""
Write-Host "=================================================="
Write-Host "  Kubeflow UI Port-Forwarding"
Write-Host "=================================================="
Write-Host ""

# Check minikube
$mkStatus = minikube status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Minikube is not running. Starting..." -ForegroundColor Yellow
    minikube start --memory=8192 --cpus=4 --driver=docker
}

Write-Host "Checking Kubeflow pods..." -ForegroundColor Cyan
kubectl get pods -n kubeflow | Select-String "ml-pipeline"

Write-Host ""
Write-Host "Starting port-forward on http://localhost:8888 ..." -ForegroundColor Green
Write-Host "Keep this window OPEN while using Kubeflow UI" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80
