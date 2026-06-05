import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df["target"] = y.values

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    return processed_df

if __name__ == "__main__":
    preprocess_data("../breast_cancer_raw.csv", "breast_cancer_preprocessing.csv")
    print("Preprocessing selesai.")
