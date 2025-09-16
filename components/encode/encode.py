# encode.py
import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    # Definir columna objetivo
    target_col = "Weather Type"

    # Identify categorical columns (excluding numerical)
    categorical_cols = [col for col in df.select_dtypes(exclude=np.number).columns if col != target_col]

    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    df.to_csv(args.out, index=False)

