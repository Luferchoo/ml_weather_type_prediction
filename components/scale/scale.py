# scale.py
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    # Identify numerical columns
    target_column = 'Weather Type' # Replace with your target column name
    numerical_cols = df.select_dtypes(include=np.number).columns.drop(target_column, errors='ignore')   


    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    df.to_csv(args.out, index=False)