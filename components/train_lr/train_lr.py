import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    parser.add_argument("--register_model", type=str, default=None, help="Nombre opcional para registrar el modelo en MLflow")
    args = parser.parse_args()

    # Activar autologging (no iniciar run manualmente)
    mlflow.sklearn.autolog()

    # Carga de datos
    train_df = pd.read_csv(args.train_data)
    target_column = 'Weather Type'

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the training data.")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Entrenamiento del modelo
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Guardado local
    model_path = os.path.join(args.model_out, "model.pkl")
    joblib.dump(model, model_path)

    # Logging del modelo en MLflow
    if args.register_model:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="logistic_regression_model",
            registered_model_name=args.register_model
        )
    else:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="logistic_regression_model"
        )
