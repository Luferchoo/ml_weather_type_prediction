import argparse
import pandas as pd
import joblib
import os
import mlflow
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",       type=str, required=True)
    parser.add_argument("--model",           type=str, required=True)
    parser.add_argument("--predictions_out", type=str, required=True)
    args, _ = parser.parse_known_args()

    mlflow.start_run()

    test_df = pd.read_csv(args.test_data)

    # Cargar modelo y feature list
    model_path = os.path.join(args.model, "model.pkl")
    model = joblib.load(model_path)

    feature_file = os.path.join(args.model, "feature_names.json")
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            feature_names = json.load(f)
    else:
        feature_names = None

    target_column = 'Weather Type'
    if target_column in test_df.columns:
        X_test = test_df.drop(columns=[target_column])
    else:
        X_test = test_df.copy()

    # Reindexar columnas si hay feature_names
    if feature_names is not None:
        for c in feature_names:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test[feature_names]

    os.makedirs("outputs", exist_ok=True)
    X_test.head(10).to_csv(os.path.join("outputs", "test_sample_head.csv"), index=False)
    mlflow.log_artifact(os.path.join("outputs", "test_sample_head.csv"), artifact_path="test_sample")

    # Predicciones
    predictions = model.predict(X_test)
    preds_df = pd.DataFrame({"predictions": predictions})

    # Probabilidades si el modelo las soporta
    try:
        proba = model.predict_proba(X_test)
        # Asegurar que todas las clases estén presentes como columnas
        wanted_classes = ["Clear", "Cloudy", "Rain", "Storm"]
        class_names = list(model.classes_)
        proba_df = pd.DataFrame(proba, columns=class_names)
        # Añadir columnas faltantes con ceros
        for cls in wanted_classes:
            if cls not in proba_df.columns:
                proba_df[cls] = 0.0
        # Reordenar columnas
        proba_df = proba_df[wanted_classes]
        preds_df = pd.concat([preds_df, proba_df], axis=1)
    except AttributeError:
        print("El modelo no soporta predict_proba; no se guardarán probabilidades.")

    # Añadir ground truth si existe
    if target_column in test_df.columns:
        preds_df[target_column] = test_df[target_column].astype(str).values

    # Guardar CSV
    os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)
    preds_df.to_csv(args.predictions_out, index=False)
    mlflow.log_artifact(args.predictions_out, artifact_path="predictions")

    mlflow.end_run()
