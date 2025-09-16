from azureml.core import Run
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import traceback

def safe_log_table(run, name, df):
    try:
        payload = df.reset_index().to_dict(orient="list")
        run.log_table(name, payload)
    except Exception as e:
        # fallback: log flat list
        try:
            flat = df.values.ravel().tolist()
            run.log_list(name + "_flat", list(map(int, flat)))
            run.log("log_table_fallback", 1)
        except Exception:
            run.log("log_table_error", str(e))
            run.log("log_table_traceback", traceback.format_exc())

def main():
    # parse_known_args para ignorar --metrics_out extra de AzureML
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    args, _ = parser.parse_known_args()

    run = Run.get_context()

    # load
    df_test = pd.read_csv(args.test_data)
    df_pred = pd.read_csv(args.predictions)

    # asegúrate que las columnas existen
    if "Weather Type" not in df_test.columns:
        raise ValueError("No se encontró columna 'Weather Type' en test_data")
    if "predictions" not in df_pred.columns:
        raise ValueError("No se encontró columna 'predictions' en predictions")

    y_true = df_test["Weather Type"].astype(str).copy()
    y_pred = df_pred["predictions"].astype(str).copy()

    # Normalización mínima: strip (quita espacios extra). Si no quieres normalizar, coméntalo.
    y_true = y_true.str.strip()
    y_pred = y_pred.str.strip()

    # 1) Diagnóstico: conteos de etiquetas
    vc_true = y_true.value_counts().to_dict()
    vc_pred = y_pred.value_counts().to_dict()
    run.log("unique_true_count", len(vc_true))
    run.log("unique_pred_count", len(vc_pred))
    # Loguear pequeños resúmenes como listas/strings
    run.log_list("labels_true_sample", list(vc_true.keys())[:50])
    run.log_list("labels_pred_sample", list(vc_pred.keys())[:50])
    run.log("value_counts_true_json", str(vc_true))
    run.log("value_counts_pred_json", str(vc_pred))

    # 2) Clases fijas en el orden que quieres ver
    wanted_classes = ["Clear", "Cloudy", "Rain", "Storm"]

    # 3) Construir matrix con la UNION de clases encontradas (para evitar errores)
    present_classes = sorted(set(y_true.unique()).union(set(y_pred.unique())))
    run.log_list("present_classes", present_classes)

    # Si detectas etiquetas que no están en wanted_classes, las registramos
    extras = [c for c in present_classes if c not in wanted_classes]
    if extras:
        run.log("extra_labels_found", str(extras))

    # 4) Calcular métricas habituales (con zero_division para evitar errores)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    run.log("accuracy", float(acc))
    run.log("precision_macro", float(prec))
    run.log("recall_macro", float(rec))
    run.log("f1_macro", float(f1))

    # 5) Matriz de confusión: la calculamos usando present_classes, luego reindexamos a wanted_classes
    try:
        cm = confusion_matrix(y_true, y_pred, labels=present_classes)
        df_cm = pd.DataFrame(cm, index=present_classes, columns=present_classes)
        # Reindex to wanted_classes rows and columns, filling missing with 0 (so shows zeros where absent)
        df_cm = df_cm.reindex(index=wanted_classes, columns=wanted_classes, fill_value=0)
    except Exception as e:
        run.log("confusion_matrix_error", str(e))
        raise

    # 6) Imagen bonita (Outputs -> Images)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_cm.values, annot=True, fmt="d", cmap="Blues",
                xticklabels=df_cm.columns, yticklabels=df_cm.index, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (reindexed to wanted classes)")
    plt.tight_layout()
    try:
        run.log_image(name="Confusion Matrix", plot=fig)
    except Exception:
        # fallback: guardar en outputs
        fig.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
        run.log("confusion_matrix_saved_to_outputs", "outputs/confusion_matrix.png")
    plt.close(fig)

    # 7) Loguear la tabla en formato aceptado por AzureML Metrics
    # Convertir DataFrame a dict de listas con la columna de labels incluida
    df_to_log = df_cm.copy()
    df_to_log.index.name = "True"
    df_to_log = df_to_log.reset_index()  # ahora la primera columna es 'True' con las etiquetas
    safe_log_table(run, "confusion_matrix_table", df_to_log)
    # al final, antes de run.complete()
    run.log("present_classes_count", len(present_classes))
    run.log("extras_count", len(extras))


    run.complete()

if __name__ == "__main__":
    main()
