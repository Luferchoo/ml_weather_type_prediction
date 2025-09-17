import argparse
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix

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

    # Predicciones y probabilidades
    predictions = model.predict(X_test)
    preds_df = pd.DataFrame({"predictions": predictions})
    
    # Log sample predictions
    sample_preds = pd.DataFrame({
        "features": [str(row) for row in X_test.iloc[:5].to_dict(orient="records")],
        "prediction": predictions[:5]
    })
    
    # Crear directorio temporal para archivos intermedios
    temp_dir = os.path.join("outputs", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Guardar predicciones de muestra en el directorio temporal
    sample_preds_path = os.path.join(temp_dir, "sample_predictions.json")
    sample_preds.to_json(sample_preds_path, orient="records", indent=2)
    mlflow.log_artifact(sample_preds_path, "sample_predictions")

    # Probabilidades si el modelo las soporta
    try:
        proba = model.predict_proba(X_test)
        # Asegurar que todas las clases estén presentes como columnas
        wanted_classes = list(model.classes_)
        proba_df = pd.DataFrame(proba, columns=wanted_classes)

        # Añadir columnas faltantes con ceros
        for cls in wanted_classes:
            if cls not in proba_df.columns:
                proba_df[cls] = 0.0
        
        # Visualizar distribución de probabilidades
        plt.figure(figsize=(10, 6))
        proba_melted = proba_df.melt(var_name='Class', value_name='Probability')
        sns.boxplot(data=proba_melted, x='Class', y='Probability')
        plt.title('Probability Distribution by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        prob_dist_plot = os.path.join(temp_dir, "probability_distribution.png")
        plt.savefig(prob_dist_plot)
        plt.close()
        mlflow.log_artifact(prob_dist_plot)

        # Log probabilidades
        preds_df = pd.concat([preds_df, proba_df], axis=1)
        
        # Log estadísticas de probabilidad
        prob_stats = {
            f"mean_prob_{cls}": float(proba_df[cls].mean())
            for cls in wanted_classes
        }
        prob_stats.update({
            f"std_prob_{cls}": float(proba_df[cls].std())
            for cls in wanted_classes
        })
        mlflow.log_metrics(prob_stats)
        
    except AttributeError:
        print("El modelo no soporta predict_proba; no se guardarán probabilidades.")

    # Añadir ground truth si existe
    if target_column in test_df.columns:
        preds_df[target_column] = test_df[target_column].astype(str).values
        y_true = test_df[target_column].values
        
        # Matriz de confusión preliminar
        cm = confusion_matrix(y_true, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Preliminary)')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        conf_matrix_plot = os.path.join(temp_dir, "confusion_matrix_preview.png")
        plt.tight_layout()
        plt.savefig(conf_matrix_plot)
        plt.close()
        mlflow.log_artifact(conf_matrix_plot)
        
        # Log algunas estadísticas básicas
        class_counts = {
            f"class_{cls}_count": int((predictions == cls).sum())
            for cls in np.unique(predictions)
        }
        mlflow.log_metrics(class_counts)

    # Guardar y registrar predicciones
    os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)
    preds_df.to_csv(args.predictions_out, index=False)
    mlflow.log_artifact(args.predictions_out, artifact_path="predictions")
    
    # Log estadísticas de predicción
    pred_stats = pd.DataFrame({
        "value_counts": preds_df["predictions"].value_counts(),
        "percentage": preds_df["predictions"].value_counts(normalize=True) * 100
    })
    pred_stats_path = os.path.join(temp_dir, "prediction_stats.csv")
    pred_stats.to_csv(pred_stats_path)
    mlflow.log_artifact(pred_stats_path)

    # End the MLflow run
    mlflow.end_run()
