import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    #1) Cargar datos
    df = pd.read_csv(args.data)

    # Seleccionamos las columnas
    target_col = "Weather Type"
    keep_cols = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
             "Cloud Cover", "Atmospheric Pressure", "UV Index", "Season",
             "Visibility (km)", "Location", "Weather Type"]
    # Assuming relevant columns are all columns in the dataset.
    # Modify this part based on your specific column selection logic.
    
    selected_cols_df = df.copy()
    selected_cols_df = df[[c for c in keep_cols if c in df.columns]].copy()

    selected_cols_df.to_csv(args.out, index=False)