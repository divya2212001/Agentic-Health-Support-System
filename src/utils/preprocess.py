import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("src/data/heart_raw.csv")
OUTPUT_PATH = Path("src/data/heart_cleaned.csv")

columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

def preprocess():
    df = pd.read_csv(DATA_PATH, header=None, names=columns)
    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    df.to_csv(OUTPUT_PATH, index=False)

    print(" Cleaned dataset saved at:", OUTPUT_PATH)
    print(df.head())
    print("\nShape:", df.shape)

if __name__ == "__main__":
    preprocess()
