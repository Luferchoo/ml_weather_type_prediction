# Azure ML SDK v2 - Titanic Pipeline (train -> score -> eval)

This project defines a 3-step pipeline using Azure ML SDK v2 with command components:

1) **train**: Trains a Logistic Regression on Titanic CSV, writes `model.pkl`.
2) **score**: Loads the model and scores a test split, writes `predictions.csv`.
3) **eval**: Computes metrics (accuracy, precision, recall, f1, roc_auc) to `metrics.json`.

> Expected input: a Titanic-style CSV with a `Survived` target column and common features.
> Adjust column names inside `components/train/train.py` if your CSV differs.

## Quick start

### 0) Requirements
- Python 3.9+
- `pip install azure-ai-ml azure-identity scikit-learn pandas numpy joblib`

### 1) Configure Azure
Set environment variables or edit `pipeline_submit.py`:
- SUBSCRIPTION_ID
- RESOURCE_GROUP
- WORKSPACE_NAME
- COMPUTE_NAME (an existing CPU cluster, e.g. `cpu-cluster`)

### 2) Put your dataset
Save your Titanic CSV at: `data/titanic.csv`

### 3) Submit pipeline
```bash
python pipeline_submit.py
```

This will:
- Load YAML components
- Build the pipeline graph
- Upload local code folders
- Submit the job to your workspace
- Print the AzureML job URL
