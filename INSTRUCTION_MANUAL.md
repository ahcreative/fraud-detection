# ══════════════════════════════════════════════════════════════════════════════
#  INSTRUCTION MANUAL
#  IEEE CIS Fraud Detection System — Complete MLOps Pipeline
#  Windows 11 | WSL | Minikube | Kubeflow | Docker Desktop
# ══════════════════════════════════════════════════════════════════════════════

## TABLE OF CONTENTS
  PART 0 — Prerequisites & Downloads
  PART 1 — Project Setup
  PART 2 — Run the Pipeline Locally (fastest path)
  PART 3 — Kubeflow Pipeline Submission
  PART 4 — Monitoring Stack (Prometheus + Grafana)
  PART 5 — CI/CD Setup
  PART 6 — Task-by-Task Mapping
  PART 7 — Troubleshooting


══════════════════════════════════════════════════════════════════════════════
PART 0 — PREREQUISITES & DOWNLOADS
══════════════════════════════════════════════════════════════════════════════

You need to install/verify the following. Everything marked ✅ you said you
already have. Install anything marked ⬜.

✅ Python 3.11+         (already installed)
✅ Docker Desktop       (already installed — keep it running)
✅ Minikube             (already installed)
✅ Kubeflow Pipelines   (already installed in Minikube)
⬜ WSL 2 Ubuntu         — needed for shell scripts

────────────────────────────────────────────────
STEP 0-A: Install WSL 2 Ubuntu (if not done)
────────────────────────────────────────────────
Open PowerShell as Administrator:

  wsl --install -d Ubuntu
  wsl --set-default-version 2

Restart your PC. Open Ubuntu from Start Menu. Create username/password.

────────────────────────────────────────────────
STEP 0-B: Install Python packages (Windows CMD or WSL)
────────────────────────────────────────────────
You can use either Windows CMD or WSL. The commands are the same.

⚠️  IMPORTANT: Use Windows CMD (not WSL) if your Kubeflow is installed
    in Minikube on Windows. This avoids path conflicts.

────────────────────────────────────────────────
STEP 0-C: Place the dataset files
────────────────────────────────────────────────
Create a folder: C:\fraud_detection\data\

Copy these 4 files into it:
  - train_transaction.csv
  - train_identity.csv
  - test_transaction.csv
  - test_identity.csv

Your data folder should look like:
  C:\fraud_detection\data\
    ├── train_transaction.csv   (~500MB)
    ├── train_identity.csv      (~30MB)
    ├── test_transaction.csv
    └── test_identity.csv


══════════════════════════════════════════════════════════════════════════════
PART 1 — PROJECT SETUP
══════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────
STEP 1-A: Extract the project ZIP
────────────────────────────────────────────────
Extract fraud_detection.zip to: C:\fraud_detection\

Your folder should look like:
  C:\fraud_detection\
    ├── pipeline/
    ├── api/
    ├── scripts/
    ├── monitoring/
    ├── kubernetes/
    ├── cicd/
    ├── drift_simulation/
    ├── explainability/
    ├── tests/
    ├── data/              ← put your CSV files here
    ├── requirements.txt
    ├── Dockerfile.training
    ├── Dockerfile.api
    ├── docker-compose.yml
    └── setup.bat

────────────────────────────────────────────────
STEP 1-B: Open Command Prompt in project folder
────────────────────────────────────────────────
1. Press Win+R, type: cmd, press Enter
2. Navigate to project:
     cd C:\fraud_detection

────────────────────────────────────────────────
STEP 1-C: Create virtual environment & install packages
────────────────────────────────────────────────
Run these commands ONE BY ONE in CMD:

  python -m venv venv
  venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt

This will take 5-10 minutes. Expected output ends with:
  Successfully installed xgboost lightgbm shap ...

Verify installation:
  python -c "import xgboost; import lightgbm; import shap; print('OK')"

────────────────────────────────────────────────
STEP 1-D: Create output directories
────────────────────────────────────────────────
  mkdir outputs
  mkdir serving
  mkdir models
  mkdir artifacts


══════════════════════════════════════════════════════════════════════════════
PART 2 — RUN THE PIPELINE LOCALLY
══════════════════════════════════════════════════════════════════════════════

This runs ALL 9 tasks without Kubeflow — great for testing on your laptop.

────────────────────────────────────────────────
STEP 2-A: Run the full local pipeline
────────────────────────────────────────────────
Make sure your venv is activated (you see "(venv)" in prompt), then:

  python scripts\run_local.py ^
    --data_dir .\data ^
    --output_dir .\outputs ^
    --imbalance_strategy smote ^
    --sample_frac 0.3

  ⚠️  --sample_frac 0.3 means use 30% of data (≈180k rows) for speed.
      This is recommended for your Core i5 + 16GB RAM.
      For full dataset: --sample_frac 1.0 (takes ~45 min)

Expected output (steps complete one by one):
  ############################################################
  #  STEP 1: Data Ingestion
  ############################################################
    Loading transaction file: .\data\train_transaction.csv
    Shape: (590540, 394)
    ...
    Merged shape: (590540, 434)
    Fraud rate : 0.0350

  ############################################################
  #  STEP 2: Data Validation
  ############################################################
    ✅ Row count OK: 177162
    ✅ All required columns present
    ...
    Status: PASS

  [... continues through all 10 steps ...]

  ##########################################################
  #  PIPELINE COMPLETE
  ##########################################################
    Total time : 18.3 minutes
    Best model : xgb_cost_sensitive
    AUC-ROC  : 0.9234
    Recall   : 0.8156

────────────────────────────────────────────────
STEP 2-B: Run imbalance strategy comparison
────────────────────────────────────────────────
(Task 2 requirement — compare SMOTE vs class_weight)

  python scripts\compare_imbalance.py ^
    --merged_data_path .\outputs\01_merged\merged_train.csv ^
    --output_dir .\outputs\imbalance_comparison ^
    --report_path .\outputs\imbalance_comparison\report.json ^
    --sample_rows 50000

Output: .\outputs\imbalance_comparison\imbalance_strategy_comparison.png

────────────────────────────────────────────────
STEP 2-C: View outputs
────────────────────────────────────────────────
All outputs are in .\outputs\:

  .\outputs\01_merged\           ← Merged CSV
  .\outputs\02_validated\        ← Validation report
  .\outputs\03_preprocessed\     ← Train/val CSVs (SMOTE applied)
  .\outputs\04_engineered\       ← Feature-engineered data
  .\outputs\05_models\           ← 5 trained model .pkl files
  .\outputs\06_evaluation\       ← Metrics, ROC curves, confusion matrices
  .\outputs\07_serving\          ← Best model ready for API
  .\outputs\08_drift\            ← Drift simulation results
  .\outputs\09_retraining\       ← Strategy comparison chart
  .\outputs\10_explainability\   ← SHAP plots
  .\outputs\imbalance_comparison\ ← Strategy comparison (Task 2)

Key files to check:
  .\outputs\06_evaluation\evaluation_report.json  ← All model metrics
  .\outputs\06_evaluation\roc_pr_curves.png        ← ROC curves
  .\outputs\06_evaluation\confusion_matrices.png   ← Confusion matrices
  .\outputs\10_explainability\shap_global_bar.png  ← Feature importance


══════════════════════════════════════════════════════════════════════════════
PART 3 — KUBEFLOW PIPELINE SUBMISSION
══════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────
STEP 3-A: Verify Kubeflow is running
────────────────────────────────────────────────
Open a NEW CMD window and run:

  minikube status

Expected output:
  minikube
  type: Control Plane
  host: Running
  kubelet: Running
  apiserver: Running
  kubeconfig: Configured

If minikube is not running:
  minikube start --memory=8192 --cpus=4 --driver=docker

Wait 3-5 minutes. Then verify Kubeflow pods:
  kubectl get pods -n kubeflow

All pods should show Running or Completed status.
(First time it takes 10-15 minutes to download images.)

────────────────────────────────────────────────
STEP 3-B: Copy data into Minikube
────────────────────────────────────────────────
Kubeflow runs inside Minikube and needs access to your data files.

  minikube ssh "sudo mkdir -p /data"
  minikube cp .\data\train_transaction.csv /data/train_transaction.csv
  minikube cp .\data\train_identity.csv /data/train_identity.csv

⚠️  The files are large. This may take 5-10 minutes.
    Alternatively, mount your data directory:
  minikube mount .\data:/data
  (Keep this CMD window open while running the pipeline)

────────────────────────────────────────────────
STEP 3-C: Apply Kubernetes resources
────────────────────────────────────────────────
  kubectl apply -f kubernetes\namespace.yaml
  kubectl apply -f kubernetes\persistent-volumes.yaml

Verify:
  kubectl get namespace fraud-detection
  kubectl get pvc -n fraud-detection

────────────────────────────────────────────────
STEP 3-D: Build Docker image inside Minikube
────────────────────────────────────────────────
Open a NEW CMD window:

  minikube docker-env --shell cmd
  (This prints SET commands — copy and run them)

Then build:
  docker build -t fraud-detection:latest -f Dockerfile.training .

────────────────────────────────────────────────
STEP 3-E: Start port-forwarding to Kubeflow UI
────────────────────────────────────────────────
Open a NEW CMD window (keep it open):

  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80

Open browser: http://localhost:8888
You should see the Kubeflow Pipelines UI.

────────────────────────────────────────────────
STEP 3-F: Compile and submit the pipeline
────────────────────────────────────────────────
In your original CMD (venv activated):

  python pipeline\pipeline.py

This creates: fraud_detection_pipeline.yaml

Then submit:
  python scripts\submit_kubeflow.py ^
    --endpoint http://localhost:8888 ^
    --data_dir /data

OR manually in the Kubeflow UI:
  1. Click "Pipelines" → "Upload pipeline"
  2. Upload fraud_detection_pipeline.yaml
  3. Click "Create run"
  4. Set parameters as needed
  5. Click "Start"

────────────────────────────────────────────────
STEP 3-G: Monitor pipeline execution
────────────────────────────────────────────────
In the Kubeflow UI (http://localhost:8888):
  - Click "Runs" to see active runs
  - Click on a run to see step-by-step execution
  - Click any step to see logs
  - Green = success, Blue = running, Red = failed

Each step has retry configured:
  - Data ingestion: 2 retries
  - Preprocessing : 1 retry
  - Training      : 1 retry


══════════════════════════════════════════════════════════════════════════════
PART 4 — MONITORING STACK (Prometheus + Grafana)
══════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────
STEP 4-A: Start the monitoring stack
────────────────────────────────────────────────
Make sure Docker Desktop is running, then:

  docker-compose up -d

Wait 30 seconds, then verify:
  docker-compose ps

All services should show "Up":
  fraud-api       Up   0.0.0.0:8000->8000/tcp
  prometheus      Up   0.0.0.0:9090->9090/tcp
  grafana         Up   0.0.0.0:3000->3000/tcp
  alertmanager    Up   0.0.0.0:9093->9093/tcp
  node-exporter   Up   0.0.0.0:9100->9100/tcp

────────────────────────────────────────────────
STEP 4-B: Copy model to API serving directory
────────────────────────────────────────────────
After running the local pipeline, copy the model to serving/:

  copy .\outputs\07_serving\model.pkl .\serving\model.pkl
  copy .\outputs\07_serving\preprocessor.pkl .\serving\preprocessor.pkl
  copy .\outputs\07_serving\feature_config.json .\serving\feature_config.json
  copy .\outputs\07_serving\deployment_metadata.json .\serving\deployment_metadata.json

Then restart the API:
  docker-compose restart fraud-api

Verify:
  curl http://localhost:8000/health

Expected: {"status":"ok","model_loaded":true,...}

────────────────────────────────────────────────
STEP 4-C: Access monitoring dashboards
────────────────────────────────────────────────
Grafana   : http://localhost:3000
            Login: admin / admin123
            Dashboards → Fraud Detection → (3 dashboards)

Prometheus: http://localhost:9090
            Try query: fraud_api_requests_total

Alertmanager: http://localhost:9093

────────────────────────────────────────────────
STEP 4-D: Test the API with sample prediction
────────────────────────────────────────────────
Send a test prediction (in CMD):

  curl -X POST http://localhost:8000/predict ^
    -H "Content-Type: application/json" ^
    -d "{\"TransactionAmt\": 117.5, \"ProductCD\": \"W\", \"card1\": 4455, \"card4\": \"visa\"}"

Expected response:
  {
    "fraud_probability": 0.0234,
    "is_fraud": false,
    "threshold": 0.5,
    "confidence": "high",
    "latency_ms": 12.3
  }

────────────────────────────────────────────────
STEP 4-E: Generate load to see metrics in Grafana
────────────────────────────────────────────────
Run this in Python to generate 100 predictions:

  python -c "
  import requests, random, time
  for i in range(100):
      r = requests.post('http://localhost:8000/predict', json={
          'TransactionAmt': random.uniform(10, 500),
          'ProductCD': random.choice(['W','H','C','S','R']),
          'card1': random.randint(1000, 9999),
      })
      print(f'{i}: {r.json()[\"fraud_probability\"]:.4f}')
      time.sleep(0.1)
  "

Then check Grafana dashboards — you'll see metrics updating.

────────────────────────────────────────────────
STEP 4-F: View alert rules in Prometheus
────────────────────────────────────────────────
1. Open http://localhost:9090
2. Click "Alerts" tab
3. You'll see rules: FraudRecallLow, HighAPILatency, etc.
4. Status shows: Inactive/Pending/Firing

To trigger a test alert, modify Prometheus rules in:
  monitoring\prometheus\alert_rules.yml


══════════════════════════════════════════════════════════════════════════════
PART 5 — CI/CD SETUP
══════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────
STEP 5-A: GitHub Actions setup
────────────────────────────────────────────────
1. Create a GitHub repository
2. Push this project to GitHub:
     git init
     git add .
     git commit -m "Initial fraud detection system"
     git remote add origin https://github.com/YOUR_USERNAME/fraud-detection.git
     git push -u origin main

3. Copy the CI/CD workflow file:
     mkdir -p .github\workflows
     copy cicd\.github\workflows\fraud_detection_cicd.yml .github\workflows\

4. Push again:
     git add .github\
     git commit -m "Add CI/CD workflow"
     git push

5. In GitHub → Settings → Secrets → Add:
     KFP_ENDPOINT = http://your-kubeflow-endpoint (if exposed publicly)

6. Go to GitHub → Actions tab → see workflow running automatically

────────────────────────────────────────────────
STEP 5-B: Trigger intelligent retraining
────────────────────────────────────────────────
To simulate monitoring-triggered retraining:
  - Go to GitHub → Actions → "Fraud Detection CI/CD Pipeline"
  - Click "Run workflow"
  - Set trigger_reason = "recall_drop"
  - Set recall_value = "0.72"
  - Click "Run workflow"

This triggers Stage 4 (intelligent retraining) in the CI/CD pipeline.

────────────────────────────────────────────────
STEP 5-C: Alertmanager → GitHub Actions webhook
────────────────────────────────────────────────
For fully automated drift-triggered retraining:

1. Create a GitHub Personal Access Token (Settings → Developer settings)
2. In alertmanager.yml, replace the webhook URL with:
     url: 'https://api.github.com/repos/USER/REPO/actions/workflows/fraud_detection_cicd.yml/dispatches'
3. Add auth headers for the GitHub API token

This is production configuration — for this project, manual workflow
dispatch demonstrates the concept.


══════════════════════════════════════════════════════════════════════════════
PART 6 — TASK-BY-TASK MAPPING
══════════════════════════════════════════════════════════════════════════════

TASK 1: Kubeflow Environment Setup
  File: kubernetes/namespace.yaml              ← Namespace + resource quotas
  File: kubernetes/persistent-volumes.yaml     ← PVCs for artifacts
  File: pipeline/pipeline.py                   ← KFP v2 pipeline definition
  Steps: data_ingestion → validation → preprocessing → feature_engineering
         → training → evaluation → deployment (conditional)
  Retry: set_retry() on each component
  Command to run: python scripts\submit_kubeflow.py

TASK 2: Data Challenges
  File: pipeline/components/data_preprocessing/preprocess.py
    - Missing values: median (V/C cols), mode (categoricals) + sentinel flags
    - High-cardinality: TargetEncoder (>10 unique) or LabelEncoder (≤10)
    - SMOTE, RandomUnderSampler, class_weight all implemented
  File: scripts/compare_imbalance.py           ← Strategy comparison
  Command: python scripts\compare_imbalance.py --merged_data_path ...

TASK 3: Model Complexity
  File: pipeline/components/model_training/train.py
    Models: xgb_standard, xgb_cost_sensitive, lgb_standard,
            lgb_cost_sensitive, hybrid_rf_xgb
  File: pipeline/components/model_evaluation/evaluate.py
    Metrics: Precision, Recall, F1, AUC-ROC, AUC-PR, Confusion Matrix

TASK 4: Cost-Sensitive Learning
  File: pipeline/components/model_training/train.py
    → scale_pos_weight = n_negatives / n_positives (both XGB and LGB)
  File: pipeline/components/model_evaluation/evaluate.py
    → business_cost = FN * $200 + FP * $5 per model
  File: scripts/compare_imbalance.py
    → Cost comparison chart

TASK 5: CI/CD Pipeline
  File: cicd/.github/workflows/fraud_detection_cicd.yml  ← GitHub Actions
  File: cicd/Jenkinsfile                                  ← Jenkins
  File: scripts/trigger_pipeline.py                       ← KFP trigger
  File: scripts/drift_analysis.py                         ← Alert handler
  Stages: Lint → Build Docker → Deploy KFP → Intelligent trigger

TASK 6: Observability & Monitoring
  File: monitoring/prometheus/prometheus.yml     ← Scrape config
  File: monitoring/prometheus/alert_rules.yml   ← 6 alert rules
  File: monitoring/prometheus/alertmanager.yml  ← Alert routing
  File: monitoring/grafana/dashboards/
    → system_health.json     (latency, throughput, errors)
    → model_performance.json (recall, FPR, fraud rate)
    → data_drift.json        (PSI trend, distribution shift)
  File: api/app.py           ← Prometheus metrics exposed at /metrics
  Command: docker-compose up -d

TASK 7: Drift Simulation
  File: drift_simulation/simulate_drift.py
    - split_temporal(): trains on earlier data, tests on later
    - inject_new_fraud_patterns(): increases fraud rate + shifts amounts
    - compute_psi() / compute_ks_statistic(): drift metrics
    - Plots: feature_drift_report.png, distribution_comparison.png
  Command: python drift_simulation\simulate_drift.py --merged_data_path ...

TASK 8: Intelligent Retraining
  File: drift_simulation/retraining_strategy.py
    - threshold_based: retrain when recall < 0.80 OR PSI > 0.15
    - periodic: retrain every 14 days
    - hybrid: periodic (30d) + emergency trigger
    - Comparison on 60-day simulation
    - Chart: retraining_strategy_comparison.png
  Command: python drift_simulation\retraining_strategy.py ...

TASK 9: Explainability
  File: explainability/shap_analysis.py
    - Global: bar chart + beeswarm (top 25 features)
    - Local: waterfall plot for individual fraud/legit transactions
    - Dependence plots: feature interaction effects
    - Standard vs cost-sensitive SHAP comparison
    - Business interpretation of top features
  Command: python explainability\shap_analysis.py ...


══════════════════════════════════════════════════════════════════════════════
PART 7 — TROUBLESHOOTING
══════════════════════════════════════════════════════════════════════════════

PROBLEM: pip install fails with "error: Microsoft Visual C++ 14.0 required"
  SOLUTION:
    1. Download "Build Tools for Visual Studio" from:
       https://visualstudio.microsoft.com/downloads/
    2. Install "C++ build tools" workload
    OR use pre-built wheels:
       pip install --only-binary=:all: lightgbm xgboost

PROBLEM: MemoryError during SMOTE or model training
  SOLUTION:
    Reduce sample fraction:
    python scripts\run_local.py --sample_frac 0.15
    OR use class_weight instead of smote (no memory overhead):
    python scripts\run_local.py --imbalance_strategy class_weight

PROBLEM: minikube start fails
  SOLUTION:
    minikube delete
    minikube start --memory=6144 --cpus=3 --driver=docker --disk-size=20g

PROBLEM: kubectl get pods shows pods in Pending state
  SOLUTION:
    kubectl describe pod <pod-name> -n kubeflow
    Usually: insufficient memory. Increase minikube memory:
    minikube stop
    minikube start --memory=8192 --cpus=4 --driver=docker

PROBLEM: port-forward connection refused (port 8888)
  SOLUTION:
    # Check if ml-pipeline-ui service exists:
    kubectl get svc -n kubeflow | grep ml-pipeline-ui
    # If it shows a different name:
    kubectl get svc -n kubeflow
    # Use the correct service name in port-forward

PROBLEM: docker-compose up fails — port already in use
  SOLUTION:
    netstat -ano | findstr :9090  (find process using port)
    taskkill /PID <PID> /F
    OR change port in docker-compose.yml (e.g., "9091:9090")

PROBLEM: SHAP runs out of memory
  SOLUTION:
    Reduce sample_size in run_local.py:
    Change: sample_size=1000 → sample_size=300
    OR: --skip_shap true (to skip SHAP entirely)

PROBLEM: train_transaction.csv not found
  SOLUTION:
    Ensure file is at: .\data\train_transaction.csv
    Verify: dir .\data\
    If in Downloads: copy C:\Users\YOU\Downloads\train_transaction.csv .\data\

PROBLEM: KFP SDK import error
  SOLUTION:
    pip uninstall kfp kfp-server-api -y
    pip install kfp==2.6.0

PROBLEM: Grafana shows "No data" in dashboards
  SOLUTION:
    1. Ensure fraud-api container is running: docker-compose ps
    2. Send some predictions (Step 4-E above)
    3. In Grafana → Explore → run: fraud_api_requests_total
    4. Check datasource: Configuration → Data Sources → Prometheus → Test

PROBLEM: GitHub Actions fails on "docker build"
  SOLUTION:
    GitHub Actions uses ubuntu-latest which may not have your local images.
    The workflow will show warnings but continue. Docker push is optional
    (needs container registry credentials).
    For local testing only, the CI lint/test stages still work.

══════════════════════════════════════════════════════════════════════════════
COLUMN TYPE REFERENCE (for your understanding)
══════════════════════════════════════════════════════════════════════════════

NUMERIC columns (continuous/ordinal values):
  TransactionDT   — seconds since reference time
  TransactionAmt  — transaction amount in USD
  card1, card2, card3, card5  — card attributes (numeric codes)
  addr1, addr2    — billing/shipping address zip codes
  dist1, dist2    — distance between addresses
  C1–C14          — count features (velocity counts per card/address)
  D1–D15          — time delta features (days since last transaction, etc.)
  V1–V339         — Vesta-engineered features (anonymized, numeric)
  id_01–id_11     — identity numeric features
  id_13, id_14, id_17–id_26, id_32  — numeric identity features

CATEGORICAL columns (text/coded categories):
  ProductCD       — product category: W, H, C, S, R
  card4           — card brand: visa, mastercard, discover, american express
  card6           — card type: credit, debit
  P_emaildomain   — purchaser email domain (gmail.com, yahoo.com, etc.)
  R_emaildomain   — recipient email domain
  M1–M9          — match flags: T (true), F (false), NaN
  id_12           — NotFound / Found
  id_15           — New / Found / Unknown
  id_16           — NotFound / Found
  id_23           — IP proxy type
  id_27–id_29     — Found/NotFound flags
  id_30           — OS version string
  id_31           — browser string
  id_33           — screen resolution
  id_34           — match_status
  id_35–id_38     — T/F flag columns
  DeviceType      — desktop / mobile
  DeviceInfo      — device OS / browser details

══════════════════════════════════════════════════════════════════════════════
QUICK COMMAND REFERENCE
══════════════════════════════════════════════════════════════════════════════

# Setup (first time only)
  python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# Run full local pipeline (recommended first step)
  python scripts\run_local.py --data_dir .\data --output_dir .\outputs --sample_frac 0.3

# Compare imbalance strategies (Task 2)
  python scripts\compare_imbalance.py --merged_data_path .\outputs\01_merged\merged_train.csv --output_dir .\outputs\imbalance_comparison --report_path .\outputs\imbalance_comparison\report.json

# Run drift simulation (Task 7)
  python drift_simulation\simulate_drift.py --merged_data_path .\outputs\01_merged\merged_train.csv --output_dir .\outputs\drift --drift_report_path .\outputs\drift\report.json

# Run retraining strategy comparison (Task 8)
  python drift_simulation\retraining_strategy.py --output_dir .\outputs\retraining --report_path .\outputs\retraining\report.json

# Run SHAP analysis (Task 9)
  python explainability\shap_analysis.py --best_model_path .\outputs\07_serving\model.pkl --val_path .\outputs\04_engineered\val_engineered.csv --models_dir .\outputs\05_models --output_dir .\outputs\shap --report_path .\outputs\shap\report.json

# Start monitoring stack
  docker-compose up -d

# Run unit tests
  pytest tests\ -v

# Submit to Kubeflow (after port-forward)
  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80   [new window]
  python scripts\submit_kubeflow.py

# Stop monitoring
  docker-compose down
