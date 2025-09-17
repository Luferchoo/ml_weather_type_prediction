from azureml.core import Run
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import traceback
import mlflow
import os

def safe_log_table(run, name, df):
    try:
        # Limita el tamaño de la tabla loggeada
        if len(df) > 10:
            df = df.head(10)
        payload = df.reset_index().to_dict(orient="list")
        run.log_table(name, payload)
    except Exception as e:
        try:
            flat = df.values.ravel().tolist()
            # Limita la lista plana también
            if len(flat) > 20:
                flat = flat[:20]
            run.log_list(name + "_flat", list(map(int, flat)))
            run.log("log_table_fallback", 1)
        except Exception:
            run.log("log_table_error", str(e))
            run.log("log_table_traceback", traceback.format_exc())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    args, _ = parser.parse_known_args()

    run = Run.get_context()
    mlflow.start_run()

    df_test = pd.read_csv(args.test_data)
    df_pred = pd.read_csv(args.predictions)

    if "Weather Type" not in df_test.columns:
        raise ValueError("No se encontró columna 'Weather Type' en test_data")
    if "predictions" not in df_pred.columns:
        raise ValueError("No se encontró columna 'predictions' en predictions")

    y_true = df_test["Weather Type"].astype(str).str.strip()
    y_pred = df_pred["predictions"].astype(str).str.strip()

    wanted_classes = ["Clear", "Cloudy", "Rain", "Storm"]

    # --- División por categoría ---
    vc_true = y_true.value_counts().reindex(wanted_classes, fill_value=0)
    vc_pred = y_pred.value_counts().reindex(wanted_classes, fill_value=0)

    # Log como tablas en AzureML
    safe_log_table(run, "class_distribution_true", vc_true.reset_index().rename(columns={"index": "class", 0: "count"}))
    safe_log_table(run, "class_distribution_pred", vc_pred.reset_index().rename(columns={"index": "class", 0: "count"}))

    # Log como métricas individuales
    for cls in wanted_classes:
        run.log(f"true_count_{cls}", int(vc_true[cls]))
        run.log(f"pred_count_{cls}", int(vc_pred[cls]))

    # Métricas principales
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    run.log("accuracy", float(acc))
    run.log("precision_macro", float(prec))
    run.log("recall_macro", float(rec))
    run.log("f1_macro", float(f1))

    # AUC-ROC solo si hay probabilidades
    if all(col in df_pred.columns for col in wanted_classes):
        y_true_bin = label_binarize(y_true, classes=wanted_classes)
        y_score = df_pred[wanted_classes].values

        # Solo calcular AUC si hay al menos dos clases presentes en y_true
        if y_true_bin.sum(axis=0).nonzero()[0].size >= 2:
            roc_auc_macro = roc_auc_score(
                y_true_bin, y_score, average="macro", multi_class="ovr"
            )
            run.log("roc_auc_ovr_macro", float(roc_auc_macro))

            for idx, cls in enumerate(wanted_classes):
                if y_true_bin[:, idx].sum() > 0:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
                    df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                    # Solo loggear primeras 10 filas para evitar error de cuota
                    safe_log_table(run, f"roc_curve_{cls}", df_roc.head(10))

                    csv_path = f"roc_curve_{cls}.csv"
                    df_roc.to_csv(csv_path, index=False)
                    mlflow.log_artifact(csv_path)

                    cls_auc = auc(fpr, tpr)
                    run.log(f"roc_auc_{cls}", float(cls_auc))
        else:
            run.log("roc_auc_ovr_macro", float("nan"))
            run.log("roc_auc_warning", "No hay suficientes clases presentes para calcular AUC-ROC.")
    else:
        run.log("roc_auc_ovr_macro", float("nan"))
        run.log("roc_auc_warning", "No se encontraron columnas de probabilidades en predictions.csv")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=wanted_classes)
    df_cm = pd.DataFrame(cm, index=wanted_classes, columns=wanted_classes)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_cm.values, annot=True, fmt="d", cmap="Blues",
                xticklabels=df_cm.columns, yticklabels=df_cm.index, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    try:
        run.log_image(name="Confusion Matrix", plot=fig)
    except Exception:
        fig.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
        run.log("confusion_matrix_saved_to_outputs", "outputs/confusion_matrix.png")

    cm_img_path = "confusion_matrix.png"
    fig.savefig(cm_img_path, bbox_inches="tight")
    mlflow.log_artifact(cm_img_path)
    plt.close(fig)

    df_to_log = df_cm.copy()
    df_to_log.index.name = "True"
    df_to_log = df_to_log.reset_index()
    safe_log_table(run, "confusion_matrix_table", df_to_log)

    cm_csv_path = "confusion_matrix_table.csv"
    df_to_log.to_csv(cm_csv_path, index=False)
    mlflow.log_artifact(cm_csv_path)

    # Muestra de resultados de clasificación
    sample_results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })
    if all(col in df_pred.columns for col in wanted_classes):
        for cls in wanted_classes:
            sample_results[f"proba_{cls}"] = df_pred[cls]
    sample_results_path = "sample_results.csv"
    sample_results.head(20).to_csv(sample_results_path, index=False)
    mlflow.log_artifact(sample_results_path)

    run.complete()
    mlflow.end_run()

if __name__ == "__main__":
    main()
