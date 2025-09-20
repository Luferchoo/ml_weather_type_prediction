import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import os
import mlflow
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--lr_predictions", type=str, required=True)
    parser.add_argument("--rf_predictions", type=str, required=True)
    parser.add_argument("--output_metrics", type=str, required=True)
    parser.add_argument("--output_report", type=str, required=True)
    parser.add_argument("--best_model_name_out", type=str, required=True) # Nueva salida
    args = parser.parse_args()

    mlflow.start_run()

    # Cargar datos
    df_test = pd.read_csv(args.test_data)
    df_lr_pred = pd.read_csv(args.lr_predictions)
    df_rf_pred = pd.read_csv(args.rf_predictions)

    target_column = 'Weather Type'
    if target_column not in df_test.columns:
        raise ValueError(f"Target column '{target_column}' not found in the test data.")

    y_true = df_test[target_column].values
    y_lr_pred = df_lr_pred["predictions"].values
    y_rf_pred = df_rf_pred["predictions"].values

    # Asegurarse de que las predicciones sean del mismo tipo que las verdaderas
    y_true = y_true.astype(str)
    y_lr_pred = y_lr_pred.astype(str)
    y_rf_pred = y_rf_pred.astype(str)

    # Calcular y registrar métricas para Logistic Regression
    lr_accuracy = accuracy_score(y_true, y_lr_pred)
    lr_precision, lr_recall, lr_f1, _ = precision_recall_fscore_support(y_true, y_lr_pred, average='weighted', zero_division=0)

    mlflow.log_metrics({
        "lr_accuracy": lr_accuracy,
        "lr_precision": lr_precision,
        "lr_recall": lr_recall,
        "lr_f1": lr_f1,
    })
    print(f"LR Accuracy: {lr_accuracy}")

    # Calcular y registrar métricas para Random Forest
    rf_accuracy = accuracy_score(y_true, y_rf_pred)
    rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_true, y_rf_pred, average='weighted', zero_division=0)

    mlflow.log_metrics({
        "rf_accuracy": rf_accuracy,
        "rf_precision": rf_precision,
        "rf_recall": rf_recall,
        "rf_f1": rf_f1,
    })
    print(f"RF Accuracy: {rf_accuracy}")

    # Determinar el mejor modelo basado en F1-score ponderado
    best_model = "LogisticRegression" if lr_f1 >= rf_f1 else "RandomForest"
    mlflow.log_param("best_model_f1", best_model)
    print(f"Best model based on F1-score: {best_model}")

    # Escribir el nombre del mejor modelo en un archivo de salida
    os.makedirs(os.path.dirname(args.best_model_name_out) or ".", exist_ok=True)
    with open(args.best_model_name_out, "w") as f:
        f.write(best_model)
    mlflow.log_artifact(args.best_model_name_out, "best_model_name")

    # Generar informe comparativo
    report_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Logistic Regression": [lr_accuracy, lr_precision, lr_recall, lr_f1],
        "Random Forest": [rf_accuracy, rf_precision, rf_recall, rf_f1]
    })

    os.makedirs(args.output_metrics, exist_ok=True)
    report_path = os.path.join(args.output_metrics, "comparison_report.csv")
    report_df.to_csv(report_path, index=False)
    mlflow.log_artifact(report_path, "comparison_report")

    # Visualización comparativa (ej. bar plot de F1-scores)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=report_df["Metric"], y=report_df["Logistic Regression"], color='skyblue', label='Logistic Regression')
    sns.barplot(x=report_df["Metric"], y=report_df["Random Forest"], color='lightcoral', label='Random Forest')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1) # Scores are typically between 0 and 1
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_report, "performance_comparison.png")
    os.makedirs(args.output_report, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path, "performance_plots")

    mlflow.end_run()

if __name__ == "__main__":
    main()
