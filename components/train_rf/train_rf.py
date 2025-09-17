# train_rf.py
import argparse
import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    parser.add_argument("--register_model", type=str, default=None)
    args = parser.parse_args()

    # autolog MLflow
    mlflow.sklearn.autolog(log_models=True, registered_model_name=args.register_model or "weather-rf")

    train_df = pd.read_csv(args.train_data)
    target_column = 'Weather Type'
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the training data.")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Guardamos nombres de columnas
    feature_names = list(X_train.columns)
    os.makedirs(args.model_out, exist_ok=True)
    feature_path = os.path.join(args.model_out, "feature_names.json")
    with open(feature_path, "w") as f:
        json.dump(feature_names, f)

    # Loguear info a MLflow
    mlflow.log_text("\n".join(feature_names), artifact_file="feature_names.txt")
    # tambien loguear sample rows for debug
    sample_csv = os.path.join(args.model_out, "train_sample.csv")
    train_df.head(10).to_csv(sample_csv, index=False)
    mlflow.log_artifact(sample_csv, artifact_path="train_sample")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # guardar modelo localmente
    model_path = os.path.join(args.model_out, "model.pkl")
    joblib.dump(model, model_path)

    # log model via mlflow
    mlflow.sklearn.log_model(sk_model=model, artifact_path="random_forest_model", registered_model_name=args.register_model if args.register_model else None)
    # log feature_names file as artifact
    mlflow.log_artifact(feature_path, artifact_path="model_features")

    # log model classes_ and some diagnostics
    mlflow.log_dict({"classes": list(map(str, model.classes_))}, "model_classes.json")
