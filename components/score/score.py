# score.py
import argparse
import pandas as pd
import joblib
import os
import mlflow
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",       type=str, required=True)
    parser.add_argument("--model",           type=str, required=True)  # carpeta del modelo
    parser.add_argument("--predictions_out", type=str, required=True)
    args, _ = parser.parse_known_args()

    mlflow.start_run()

    test_df = pd.read_csv(args.test_data)

    # cargar modelo y feature list
    model_path = os.path.join(args.model, "model.pkl")
    model = joblib.load(model_path)

    feature_file = os.path.join(args.model, "feature_names.json")
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            feature_names = json.load(f)
    else:
        feature_names = None

    target_column = 'Weather Type'
    if target_column not in test_df.columns:
        # no hay label en test: loguearlo y seguir (pero evaluate necesitará el ground truth)
        mlflow.log_param("test_has_ground_truth", False)
        X_test = test_df.copy()
    else:
        mlflow.log_param("test_has_ground_truth", True)
        y_test = test_df[target_column].astype(str).copy()
        X_test = test_df.drop(columns=[target_column])

    # registrar columnas de test y compararlas
    mlflow.log_text("\n".join(list(X_test.columns)), artifact_file="test_columns.txt")

    # Si existe feature_names, reindexar/reordenar (y rellenar con 0 si faltan columnas)
    if feature_names is not None:
        # Avisar sobre columnas faltantes/extra
        missing = [c for c in feature_names if c not in X_test.columns]
        extra   = [c for c in X_test.columns if c not in feature_names]
        mlflow.log_dict({"missing_features_in_test": missing, "extra_features_in_test": extra}, "feature_alignment.json")

        # Reindex safely: si faltan columnas, crear columna con zeros
        for c in feature_names:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test[feature_names]  # reorder
    else:
        # no hay feature list guardada: log warning
        mlflow.log_param("feature_list_available", False)

    # Log a sample for debugging
    sample_path = os.path.join("outputs", "test_sample_head.csv")
    os.makedirs("outputs", exist_ok=True)
    X_test.head(10).to_csv(sample_path, index=False)
    mlflow.log_artifact(sample_path, artifact_path="test_sample")

    # predict
    predictions = model.predict(X_test)
    preds_df = pd.DataFrame({"predictions": predictions})
    # si teníamos truth, inclúyela en el CSV para facilitar compare en evaluate
    if target_column in test_df.columns:
        preds_df[target_column] = test_df[target_column].astype(str).values

    # guardar predicciones
    os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)
    preds_df.to_csv(args.predictions_out, index=False)
    mlflow.log_artifact(args.predictions_out, artifact_path="predictions")

    # log model classes
    try:
        mlflow.log_dict({"classes": list(map(str, model.classes_))}, "model_classes_from_score.json")
    except Exception:
        pass

    mlflow.end_run()
