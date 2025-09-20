# split.py
import argparse
import pandas as pd
import os # Importar os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--train_out", type=str)
    parser.add_argument("--test_out", type=str)
    args = parser.parse_args()

    # Construir la ruta completa al archivo CSV dentro del directorio de entrada
    input_csv_path = os.path.join(args.data, "processed_data_with_sentiment.csv")
    df = pd.read_csv(input_csv_path)
    target_column = 'Weather Type' 
    #print(f"Columns in the dataset: {df.columns.tolist()}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)