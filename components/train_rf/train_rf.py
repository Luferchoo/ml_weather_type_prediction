# train_rf.py
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    parser.add_argument("--register_model", type=str, default=None)
    args = parser.parse_args()

    # Iniciar MLflow run y activar autologging
    mlflow.start_run()
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True,
        registered_model_name=args.register_model or "weather-rf"
    )

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

    # Entrenar modelo con parámetros para evitar overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,           # Limitar profundidad de los árboles
        min_samples_split=5,    # Mínimo de muestras para dividir un nodo
        min_samples_leaf=2,     # Mínimo de muestras en hojas
        max_features='sqrt',    # Usar sqrt(n_features) features en cada split
        random_state=42,
        oob_score=True         # Activar out-of-bag score para validación
    )
    model.fit(X_train, y_train)
    
    # Calcular y loguear OOB score
    oob_score = model.oob_score_
    mlflow.log_metric("oob_score", float(oob_score))

    # Calcular y registrar métricas de entrenamiento
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average='weighted'
    )

    # Preparar métricas asegurando que no hay valores None
    metrics_dict = {
        "train_accuracy": float(train_accuracy),
        "train_precision": float(train_precision),
        "train_recall": float(train_recall),
        "train_f1": float(train_f1),
        "n_estimators": float(model.n_estimators)
    }
    
    # Añadir max_depth solo si no es None
    max_depth = model.get_params().get("max_depth")
    if max_depth is not None:
        metrics_dict["max_depth"] = float(max_depth)
    
    mlflow.log_metrics(metrics_dict)

    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Visualización de importancia de características
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importance - Random Forest')
    plt.tight_layout()
    importance_plot = os.path.join(args.model_out, "feature_importance.png")
    plt.savefig(importance_plot)
    plt.close()
    mlflow.log_artifact(importance_plot, "feature_importance")

    # Guardar feature importance como CSV
    importance_csv = os.path.join(args.model_out, "feature_importance.csv")
    feature_importance.to_csv(importance_csv, index=False)
    mlflow.log_artifact(importance_csv, "feature_importance")

    # Sample predictions y probabilidades
    sample_preds = model.predict(X_train[:5])
    sample_proba = model.predict_proba(X_train[:5])
    mlflow.log_dict({
        "sample_predictions": {
            "features": X_train[:5].to_dict(orient="records"),
            "predictions": sample_preds.tolist(),
            "probabilities": sample_proba.tolist()
        }
    }, "sample_predictions.json")

    # Guardar modelo localmente
    model_path = os.path.join(args.model_out, "model.pkl")
    joblib.dump(model, model_path)

    # Log model vía MLflow con firma y ejemplo
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model",
        registered_model_name=args.register_model if args.register_model else None,
        input_example=X_train.iloc[:5]
    )

    # Log feature_names y otros metadatos
    mlflow.log_artifact(feature_path, artifact_path="model_features")
    mlflow.log_dict({
        "classes": list(map(str, model.classes_)),
        "n_features": len(feature_names),
        "model_parameters": model.get_params(),
        "oob_score": getattr(model, "oob_score_", None),
        "n_estimators": model.n_estimators
    }, "model_metadata.json")
