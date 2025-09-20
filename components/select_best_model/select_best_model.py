import argparse
import os
import shutil
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model_name_in", type=str, required=True)
    parser.add_argument("--lr_model_path", type=str, required=True)
    parser.add_argument("--rf_model_path", type=str, required=True)
    parser.add_argument("--selected_model_output", type=str, required=True)
    args = parser.parse_args()

    mlflow.start_run()

    # Leer el nombre del mejor modelo
    with open(args.best_model_name_in, "r") as f:
        best_model_name = f.read().strip()

    print(f"El mejor modelo seleccionado es: {best_model_name}")

    # Determinar qué modelo copiar
    source_model_path = ""
    if best_model_name == "LogisticRegression":
        source_model_path = args.lr_model_path
    elif best_model_name == "RandomForest":
        source_model_path = args.rf_model_path
    else:
        raise ValueError(f"Nombre de modelo desconocido: {best_model_name}")

    # Copiar el mejor modelo a la ruta de salida
    os.makedirs(args.selected_model_output, exist_ok=True)
    
    # Asegurarse de que el directorio de destino está vacío o eliminar su contenido
    for item in os.listdir(args.selected_model_output):
        item_path = os.path.join(args.selected_model_output, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    # Copiar contenido del directorio fuente al directorio de destino
    for item in os.listdir(source_model_path):
        s = os.path.join(source_model_path, item)
        d = os.path.join(args.selected_model_output, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=True) # Usar symlinks=True para evitar problemas de permisos/copia profunda
        else:
            shutil.copy2(s, d)

    print(f"Modelo '{best_model_name}' copiado a {args.selected_model_output}")
    mlflow.log_artifact(args.selected_model_output, "selected_model")

    mlflow.end_run()

if __name__ == "__main__":
    main()
