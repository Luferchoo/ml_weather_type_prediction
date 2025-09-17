# Copilot Instructions for weather-pipeline

## Project Overview
- This repo defines a modular ML pipeline for weather (Titanic-style) classification using Azure ML SDK v2.
- Pipeline steps are implemented as reusable command components in `components/`, each with a `.py` (logic) and `.yml` (AzureML spec).
- The main entrypoint is `pipeline_submit.py`, which loads YAMLs, builds the pipeline, and submits to AzureML.

## Key Directories & Files
- `components/`: Each subfolder is a pipeline step (e.g., `train_lr`, `score`, `eval`, etc.).
  - Each step: `step.py` (code), `step.yml` (AzureML component spec)
- `data/`: Input CSV data (default: `weather_classification_data.csv`)
- `envs/conda.yml`: Conda environment for AzureML jobs
- `pipeline_submit.py`: Pipeline definition and submission script

## Developer Workflows
- **Install dependencies:** Use Python 3.9+ and install packages from README.
- **Run pipeline locally:** `python pipeline_submit.py` (requires AzureML config)
- **Add new step:**
  1. Create a new folder in `components/` with `step.py` and `step.yml`.
  2. Register the component in `pipeline_submit.py`.
- **Data format:** Input CSV must match expected columns (see `train_lr/train_lr.py`).

## Patterns & Conventions
- Each pipeline step is isolated: no cross-imports between `components/`.
- All data exchange between steps is via files (CSV, pickle, JSON), not in-memory.
- Component YAMLs define inputs/outputs and environment.
- Use only standard Python, scikit-learn, pandas, numpy, joblib in component code.
- No hardcoded Azure credentials; use env vars or edit `pipeline_submit.py`.

## Example: Adding a New Model
- Copy `components/train_lr` to `components/train_newmodel`.
- Update `train_newmodel.py` for your model logic.
- Edit `train_newmodel.yml` for inputs/outputs.
- Register in `pipeline_submit.py`.

## Troubleshooting
- If pipeline fails, check AzureML job logs (URL printed by `pipeline_submit.py`).
- Ensure your input CSV matches expected schema.
- For environment issues, update `envs/conda.yml` and re-register components.

---
For more, see `README.md` and component folders for concrete examples.
