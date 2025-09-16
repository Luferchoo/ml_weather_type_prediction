from azureml.core import Run
import argparse, pandas as pd, json, os, tempfile, matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

import mlflow, mlflow.sklearn

def main():
    # Contextos de Azure ML y MLflow
    run = Run.get_context()
    mlflow.start_run()

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",   type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, required=True)
    args = parser.parse_args()

    # Autologging (solo para artefactos, no interfiere con run.log)
    mlflow.sklearn.autolog()

    # 1) Carga de datos
    df_test  = pd.read_csv(args.test_data)
    df_pred  = pd.read_csv(args.predictions)
    y_true   = df_test["Weather Type"]
    y_pred   = df_pred["predictions"]

    # 2) Métricas globales
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")
    f1   = f1_score(y_true, y_pred, average="macro")

    run.log("accuracy",        float(acc))
    run.log("precision_macro", float(prec))
    run.log("recall_macro",    float(rec))
    run.log("f1_macro",        float(f1))

    # 3) Coordenadas para ROC y PR (micro-average)
    classes    = sorted(y_true.unique())
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc     = auc(fpr, tpr)
    run.log("roc_auc_micro", float(roc_auc))

    run.log_list("roc_fpr",  [float(x) for x in fpr])
    run.log_list("roc_tpr",  [float(y) for y in tpr])

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_true_bin.ravel(), y_pred_bin.ravel()
    )
    run.log_list("pr_recall",    [float(x) for x in recall_vals])
    run.log_list("pr_precision", [float(x) for x in precision_vals])

    # 4) Matriz de confusión como listas de métricas
    #    Cada fila i corresponde a los counts de la clase true = classes[i]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    for i, cls in enumerate(classes):
        key = f"cm_true_{cls}"
        values = [float(v) for v in cm[i]]
        run.log_list(key, values)

    # 5) Guardar CSV/JSON via MLflow
    metrics = {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "roc_auc_micro": roc_auc
    }
    pd.DataFrame([metrics]).to_csv(args.metrics_out, index=False)
    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(args.metrics_out, artifact_path="metrics_csv")
        mlflow.log_artifact(json_path,     artifact_path="metrics_json")

    mlflow.end_run()
    run.complete()

if __name__ == "__main__":
    main()
