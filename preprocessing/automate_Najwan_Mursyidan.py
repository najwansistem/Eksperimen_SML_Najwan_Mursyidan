from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path=None, output_path=None):
    """Melakukan preprocessing otomatis dan mengembalikan data siap latih."""
    base_dir = Path(__file__).resolve().parent
    input_path = Path(input_path) if input_path else base_dir.parent / "breast_cancer_raw.csv"
    output_path = Path(output_path) if output_path else base_dir / "breast_cancer_preprocessing.csv"

    df = pd.read_csv(input_path)
    if "diagnosis_label" in df.columns:
        df = df.drop(columns=["diagnosis_label"])

    if "target" not in df.columns:
        raise ValueError("Kolom target tidak ditemukan pada dataset.")

    feature_columns = [col for col in df.columns if col != "target"]
    X = df[feature_columns]
    y = df["target"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=feature_columns)
    processed_df["target"] = y.values
    processed_df.to_csv(output_path, index=False)
    return processed_df


if __name__ == "__main__":
    data = preprocess_data()
    print("Preprocessing selesai.")
    print("Shape:", data.shape)
    print("Output: breast_cancer_preprocessing.csv")