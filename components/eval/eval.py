import argparse
import pandas as pd
import json
import os
import tempfile
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    mlflow.start_run()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--predictions", type=str)
    parser.add_argument("--metrics_out", type=str)
    args = parser.parse_args()

    # Activar autologging para evitar conflictos con Azure ML
    mlflow.sklearn.autolog()

    # Cargar datos
    test_df = pd.read_csv(args.test_data)
    predictions_df = pd.read_csv(args.predictions)

    target_column = 'Weather Type'
    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the test data.")

    y_true = test_df[target_column]
    y_pred = predictions_df['predictions']

    # Calcular métricas multiclase
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_macro", precision)
    mlflow.log_metric("recall_macro", recall)
    mlflow.log_metric("f1_macro", f1)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # Guardar métricas como CSV en la ruta oficial
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(args.metrics_out, index=False)

    # Crear carpeta temporal para artefactos adicionales
    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = os.path.join(temp_dir, "metrics.json")

        # Guardar JSON
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Logging de artefactos en MLflow
        mlflow.log_artifact(args.metrics_out, artifact_path="metrics")
        mlflow.log_artifact(json_path, artifact_path="metrics_json")
    mlflow.end_run()
