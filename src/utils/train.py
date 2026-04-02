import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_and_save(file_path="src/data/heart_cleaned.csv"):

    df = pd.read_csv(file_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

    os.makedirs("models", exist_ok=True)
    all_metrics = {}

    for name, pipeline in models.items():

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        all_metrics[name] = metrics

        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(pipeline, os.path.join("models", filename))

    joblib.dump(all_metrics, "models/model_metrics.pkl")

    print("Models and metrics saved successfully.")


if __name__ == "__main__":
    train_and_save()
