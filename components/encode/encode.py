# encode.py
import argparse
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    # Definir columna objetivo
    target_col = "Weather Type"

    # 1. Codificar la variable objetivo
    if target_col in df.columns:
        le = LabelEncoder()
        # Guardar los valores originales para referencia
        original_classes = df[target_col].unique()
        # Codificar la variable objetivo
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        
        # Guardar el encoder y el mapping
        encoder_dir = os.path.dirname(args.out)
        joblib.dump(le, os.path.join(encoder_dir, "label_encoder.pkl"))
        
        # Guardar el mapping para referencia
        mapping = {
            "original_classes": list(le.classes_),
            "encoded_values": le.transform(le.classes_).tolist()
        }
        with open(os.path.join(encoder_dir, "label_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)

    # 2. Codificar variables categóricas (excluyendo el target)
    categorical_cols = [col for col in df.select_dtypes(exclude=np.number).columns if col != target_col]

    if len(categorical_cols) > 0:
        # One-hot encoding para variables categóricas
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_data, 
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        
        # Combinar con el dataframe original
        df = pd.concat([
            df.drop(columns=categorical_cols), 
            encoded_df
        ], axis=1)
        
        # Guardar el one-hot encoder
        joblib.dump(encoder, os.path.join(encoder_dir, "onehot_encoder.pkl"))
        
        # Guardar metadata del one-hot encoding
        onehot_mapping = {
            "categorical_columns": categorical_cols,
            "encoded_features": encoder.get_feature_names_out(categorical_cols).tolist()
        }
        with open(os.path.join(encoder_dir, "onehot_mapping.json"), "w") as f:
            json.dump(onehot_mapping, f, indent=2)

    # Guardar el dataframe transformado
    df.to_csv(args.out, index=False)

