"""
Component Dockerfiles helper — generates individual component Dockerfiles
for Kubeflow pipeline components that need separate images.
For this project we use a single base image (fraud-detection:latest)
for all components to keep things simple.
"""

COMPONENT_BASE_IMAGE = "fraud-detection:latest"

# Component-to-script mapping for reference
COMPONENT_MAP = {
    "data_ingestion": {
        "script": "pipeline/components/data_ingestion/ingest.py",
        "entrypoint": "python pipeline/components/data_ingestion/ingest.py",
        "resources": {"cpu": "1", "memory": "4G"},
        "retry": 2,
    },
    "data_validation": {
        "script": "pipeline/components/data_validation/validate.py",
        "entrypoint": "python pipeline/components/data_validation/validate.py",
        "resources": {"cpu": "500m", "memory": "2G"},
        "retry": 1,
    },
    "preprocessing": {
        "script": "pipeline/components/data_preprocessing/preprocess.py",
        "entrypoint": "python pipeline/components/data_preprocessing/preprocess.py",
        "resources": {"cpu": "2", "memory": "6G"},
        "retry": 1,
    },
    "feature_engineering": {
        "script": "pipeline/components/feature_engineering/engineer.py",
        "entrypoint": "python pipeline/components/feature_engineering/engineer.py",
        "resources": {"cpu": "1", "memory": "4G"},
        "retry": 0,
    },
    "model_training": {
        "script": "pipeline/components/model_training/train.py",
        "entrypoint": "python pipeline/components/model_training/train.py",
        "resources": {"cpu": "2", "memory": "8G"},
        "retry": 1,
    },
    "model_evaluation": {
        "script": "pipeline/components/model_evaluation/evaluate.py",
        "entrypoint": "python pipeline/components/model_evaluation/evaluate.py",
        "resources": {"cpu": "2", "memory": "4G"},
        "retry": 0,
    },
    "deployment": {
        "script": "pipeline/components/deployment/deploy.py",
        "entrypoint": "python pipeline/components/deployment/deploy.py",
        "resources": {"cpu": "500m", "memory": "1G"},
        "retry": 0,
    },
}
