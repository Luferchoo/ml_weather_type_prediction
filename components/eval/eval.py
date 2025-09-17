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
import numpy as np
import time
import uuid

def generate_unique_filename(base_name, extension):
    """Genera un nombre de archivo único usando UUID."""
    unique_id = str(uuid.uuid4())[:8]  # Usamos los primeros 8 caracteres del UUID
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}_{unique_id}.{extension}"

def safe_log_metric(run, name, value):
    """Registra una métrica de forma segura tanto en Azure ML Run como en MLflow."""
    try:
        # Validar y convertir el valor
        if value is None or np.isnan(value):
            print(f"Valor no válido para métrica {name}: {value}")
            return False
            
        # Convertir a float y verificar que es un número válido
        float_value = float(value)
        if not np.isfinite(float_value):
            print(f"Valor infinito para métrica {name}: {float_value}")
            return False
            
        # Registrar en Azure ML Run
        try:
            run.log(name, float_value)
        except Exception as e:
            print(f"Error al registrar en Azure ML Run - {name}: {str(e)}")
        
        # Registrar en MLflow
        try:
            mlflow.log_metric(name, float_value)
        except Exception as e:
            print(f"Error al registrar en MLflow - {name}: {str(e)}")
            
        return True
    except Exception as e:
        print(f"Error general al registrar métrica {name}: {str(e)}")
        return False

def safe_log_table(run, name, df, timestamp=None):
    """Registra una tabla de forma segura en Azure ML Run y guarda un CSV para MLflow."""
    try:
        # Crear directorio para archivos temporales si no existe
        os.makedirs("outputs", exist_ok=True)
        
        # Limitar tamaño y convertir a string
        if len(df) > 10:
            df = df.head(10)
        df = df.astype(str)
        
        # Guardar como CSV para MLflow con timestamp único
        timestamp = timestamp or int(time.time())
        csv_path = os.path.join("outputs", f"{name}_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        # Intentar registrar en Azure ML Run
        try:
            payload = df.reset_index().to_dict(orient="list")
            run.log_table(name, payload)
        except Exception as e:
            print(f"Error al registrar tabla en Azure ML Run - {name}: {str(e)}")
            
    except Exception as e:
        print(f"Error al registrar tabla {name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    args, _ = parser.parse_known_args()

    # Generar timestamp único para este run
    timestamp = int(time.time())

    # Obtener el contexto de AzureML fuera del try para manejar errores
    try:
        run = Run.get_context()
    except Exception as e:
        print(f"Error obteniendo contexto de AzureML: {str(e)}")
        run = None
    
    # Asegurar que el run de MLflow se cierre correctamente
    with mlflow.start_run():
        df_test = pd.read_csv(args.test_data)
        df_pred = pd.read_csv(args.predictions)

    if "Weather Type" not in df_test.columns:
        raise ValueError("No se encontró columna 'Weather Type' en test_data")
    if "predictions" not in df_pred.columns:
        raise ValueError("No se encontró columna 'predictions' en predictions")

    # Validar y convertir etiquetas a numéricas
    try:
        y_true = df_test["Weather Type"].astype(int)
        y_pred = df_pred["predictions"].astype(int)
        
        # Verificar que no hay valores nulos o inválidos
        if y_true.isnull().any() or y_pred.isnull().any():
            print("Advertencia: Se encontraron valores nulos en las etiquetas")
            y_true = y_true.fillna(-1).astype(int)
            y_pred = y_pred.fillna(-1).astype(int)
        
        # Obtener las clases únicas presentes en los datos
        unique_classes = sorted(set(y_true) | set(y_pred) - {-1})  # Excluir valores de relleno
        wanted_classes = list(range(len(unique_classes)))
        
        print(f"Clases encontradas: {unique_classes}")
        print(f"Número de muestras: {len(y_true)}")
    except Exception as e:
        print(f"Error al procesar etiquetas: {str(e)}")
        raise

    # --- División por categoría ---
    try:
        # Conteo de valores con manejo de errores
        vc_true = y_true.value_counts().reindex(wanted_classes, fill_value=0)
        vc_pred = y_pred.value_counts().reindex(wanted_classes, fill_value=0)

        # Preparar DataFrames asegurando tipos de datos válidos
        df_true = vc_true.reset_index().rename(columns={"index": "class_id", 0: "count"})
        df_pred = vc_pred.reset_index().rename(columns={"index": "class_id", 0: "count"})
        
        # Convertir a tipo string para evitar problemas de serialización
        df_true = df_true.astype(str)
        df_pred = df_pred.astype(str)

        # Log como tablas en AzureML usando timestamp
        safe_log_table(run, "class_distribution_true", df_true, timestamp)
        safe_log_table(run, "class_distribution_pred", df_pred, timestamp)        # Log como métricas individuales con validación
        for cls in wanted_classes:
            try:
                true_count = int(vc_true[cls])
                pred_count = int(vc_pred[cls])
                
                # Verificar que los valores son válidos
                if not np.isnan(true_count):
                    run.log(f"true_count_class_{cls}", true_count)
                if not np.isnan(pred_count):
                    run.log(f"pred_count_class_{cls}", pred_count)
            except (ValueError, TypeError) as e:
                print(f"Error al registrar conteos para clase {cls}: {str(e)}")
                continue
    except Exception as e:
        print(f"Error al procesar distribución de clases: {str(e)}")
        # Continuar con la ejecución

    # Métricas principales
    try:
        # Calcular y registrar métricas globales una a una
        acc = accuracy_score(y_true, y_pred)
        safe_log_metric(run, "accuracy", acc)

        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        safe_log_metric(run, "precision_macro", prec_macro)

        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        safe_log_metric(run, "recall_macro", rec_macro)

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        safe_log_metric(run, "f1_macro", f1_macro)
        
        # Calcular métricas por clase con manejo de errores
        try:
            prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            # Registrar métricas por clase una a una
            for i, cls in enumerate(wanted_classes):
                if i < len(prec_per_class) and i < len(rec_per_class) and i < len(f1_per_class):
                    if not np.isnan(prec_per_class[i]):
                        safe_log_metric(run, f"precision_class_{cls}", float(prec_per_class[i]))
                    if not np.isnan(rec_per_class[i]):
                        safe_log_metric(run, f"recall_class_{cls}", float(rec_per_class[i]))
                    if not np.isnan(f1_per_class[i]):
                        safe_log_metric(run, f"f1_class_{cls}", float(f1_per_class[i]))
        except Exception as e:
            print(f"Error calculando métricas por clase: {str(e)}")
    except Exception as e:
        print(f"Error registrando métricas: {str(e)}")
        # Asegurar que el run termine correctamente incluso si hay error en métricas
        pass

    # AUC-ROC solo si hay probabilidades
    try:
        if all(col in df_pred.columns for col in wanted_classes):
            try:
                # Preparar datos para ROC
                y_true_bin = label_binarize(y_true, classes=wanted_classes)
                y_score = df_pred[wanted_classes].values.astype(float)
                y_score = np.nan_to_num(y_score)  # Reemplazar NaN con 0

                # Calcular y registrar ROC-AUC global
                if y_true_bin.sum(axis=0).nonzero()[0].size >= 2:
                    try:
                        roc_auc_macro = roc_auc_score(
                            y_true_bin, y_score, average="macro", multi_class="ovr"
                        )
                        safe_log_metric(run, "roc_auc_ovr_macro", roc_auc_macro)
                    except Exception as e:
                        print(f"Error en ROC-AUC macro: {str(e)}")

                    # ROC por clase
                    for idx, cls in enumerate(wanted_classes):
                        if y_true_bin[:, idx].sum() > 0:
                            try:
                                # Calcular curva ROC
                                fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
                                df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                                
                                # Guardar datos de la curva con timestamp
                                os.makedirs("outputs", exist_ok=True)
                                csv_path = os.path.join("outputs", f"roc_curve_{cls}_{timestamp}.csv")
                                df_roc.to_csv(csv_path, index=False)
                                mlflow.log_artifact(csv_path)

                                # Registrar AUC por clase
                                cls_auc = auc(fpr, tpr)
                                safe_log_metric(run, f"roc_auc_{cls}", cls_auc)
                            except Exception as e:
                                print(f"Error en ROC clase {cls}: {str(e)}")
                else:
                    print("Insuficientes clases para ROC-AUC")
            except Exception as e:
                print(f"Error en preparación ROC: {str(e)}")
        else:
            print("No hay columnas de probabilidad")
    except Exception as e:
        print(f"Error en ROC-AUC: {str(e)}")
        # Continuar ejecución

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=wanted_classes)
    df_cm = pd.DataFrame(cm, index=wanted_classes, columns=wanted_classes)
    
    # Registrar métricas de la matriz de confusión
    try:
        n_samples = float(np.sum(cm))
        if n_samples > 0:
            for i in range(len(wanted_classes)):
                for j in range(len(wanted_classes)):
                    try:
                        count = int(cm[i, j])
                        if count >= 0:  # Solo registrar valores no negativos
                            safe_log_metric(run, f"conf_matrix_count_{i}_{j}", count)
                            
                            pct = (float(count) / n_samples) * 100
                            if 0 <= pct <= 100:  # Validar porcentaje
                                safe_log_metric(run, f"conf_matrix_pct_{i}_{j}", pct)
                    except Exception as e:
                        print(f"Error en celda [{i},{j}]: {str(e)}")
        else:
            print("Matriz de confusión vacía")
    except Exception as e:
        print(f"Error en matriz de confusión: {str(e)}")
        # Continuar ejecución

    # Visualización de la matriz de confusión
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_cm.values, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Class {i}" for i in df_cm.columns],
                yticklabels=[f"Class {i}" for i in df_cm.index],
                ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    
    # Guardar y registrar la visualización
    # Guardar y registrar la visualización con nombre único
    os.makedirs("outputs", exist_ok=True)
    cm_img_filename = generate_unique_filename("confusion_matrix", "png")
    cm_img_path = os.path.join("outputs", cm_img_filename)
    
    try:
        run.log_image(name="Confusion Matrix", plot=fig)
    except Exception:
        fig.savefig(cm_img_path, bbox_inches="tight")
        run.log("confusion_matrix_saved_to_outputs", cm_img_path)
    
    # Guardar para MLflow con nombre único
    fig.savefig(cm_img_path, bbox_inches="tight")
    mlflow.log_artifact(cm_img_path)
    plt.close(fig)
    
    df_to_log = df_cm.copy()
    df_to_log.index.name = "True"
    df_to_log = df_to_log.reset_index()
    
    # Usar nombre único para la tabla de confusión
    cm_csv_filename = generate_unique_filename("confusion_matrix_table", "csv")
    cm_csv_path = cm_csv_filename  # Para mantener compatibilidad con el código siguiente
    os.makedirs("outputs", exist_ok=True)
    cm_csv_path_full = os.path.join("outputs", cm_csv_path)
    df_to_log.to_csv(cm_csv_path_full, index=False)
    mlflow.log_artifact(cm_csv_path_full)

    # Muestra de resultados de clasificación
    sample_results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })
    if all(col in df_pred.columns for col in wanted_classes):
        for cls in wanted_classes:
            sample_results[f"proba_{cls}"] = df_pred[cls]
    # Guardar resultados con nombre único
    sample_results_filename = generate_unique_filename("sample_results", "csv")
    sample_results_path = os.path.join("outputs", sample_results_filename)
    sample_results.head(20).to_csv(sample_results_path, index=False)
    mlflow.log_artifact(sample_results_path)

    # Asegurarse de que el run de AzureML se complete correctamente
    if run is not None:
        try:
            run.complete()
        except Exception as e:
            print(f"Error al completar run de AzureML: {str(e)}")
    # El run de MLflow se cierra automáticamente por el contexto with

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error en la ejecución principal: {str(e)}")
        raise
