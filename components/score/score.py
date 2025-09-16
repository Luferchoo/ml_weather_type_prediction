import argparse
import pandas as pd
import joblib
import os
import mlflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",       type=str)
    parser.add_argument("--model",           type=str)
    parser.add_argument("--predictions_out", type=str)
    args = parser.parse_args()

    # Inicia el MLflow Run
    mlflow.start_run()

    test_df = pd.read_csv(args.test_data)

    # Cargar el modelo
    model_path = os.path.join(args.model, "model.pkl")
    model = joblib.load(model_path)

    target_column = 'Weather Type'
    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the test data.")

    X_test = test_df.drop(columns=[target_column])
    predictions = model.predict(X_test)

    predictions_df = pd.DataFrame(predictions, columns=['predictions'])
    predictions_df.to_csv(args.predictions_out, index=False)

    # Log del CSV de predicciones
    mlflow.log_artifact(args.predictions_out, artifact_path="predictions")

    # Finaliza el MLflow Run
    mlflow.end_run()
