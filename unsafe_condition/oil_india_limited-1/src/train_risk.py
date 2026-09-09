import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    EMBEDDINGS_FILE,
    MODELS_DIR,
    RISK_LABEL_ENCODER_FILE,
    RISK_MODEL_FILE,
    STANDARDIZED_FILE,
)


VALID_RISK_LEVELS = ["Low", "Medium", "High", "Critical"]


def main():
    print("Loading standardized risk data...")
    df = pd.read_csv(STANDARDIZED_FILE)

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)

    print(f"Total reports: {len(df)}")
    print(f"Embedding features per report: {embeddings.shape[1]}")

    if len(df) != len(embeddings):
        raise ValueError(
            f"Data mismatch: standardized CSV has {len(df)} rows, "
            f"but embeddings file has {len(embeddings)} vectors."
        )

    if "risk_level" not in df.columns:
        raise ValueError(
            "Column 'risk_level' was not found in "
            f"{STANDARDIZED_FILE}.\n"
            "Your standardized dataset must contain one of: "
            "Low, Medium, High, Critical."
        )

    df["risk_level"] = (
        df["risk_level"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )

    valid_rows = df["risk_level"].isin(VALID_RISK_LEVELS)

    df = df.loc[valid_rows].reset_index(drop=True)
    embeddings = embeddings[valid_rows.to_numpy()]

    if len(df) == 0:
        raise ValueError(
            "No valid risk levels were found. Expected: "
            "Low, Medium, High, Critical."
        )

    X = embeddings
    risk_labels = df["risk_level"]

    print("\nRisk-level distribution:")
    print(risk_labels.value_counts())

    if risk_labels.nunique() < 2:
        raise ValueError(
            "Risk model training requires at least two different risk levels."
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(risk_labels)

    print("\nRisk class mapping:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"{index} = {label}")

    class_counts = pd.Series(y).value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Every risk level needs at least 2 records for a stratified train-test split.\n"
            f"Current counts:\n{risk_labels.value_counts()}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining risk-level model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).astype(int)

    print("\n--- Risk Model Evaluation ---")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(label_encoder.classes_))),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, RISK_MODEL_FILE)
    joblib.dump(label_encoder, RISK_LABEL_ENCODER_FILE)

    print(f"\nSaved risk model: {RISK_MODEL_FILE}")
    print(f"Saved risk label encoder: {RISK_LABEL_ENCODER_FILE}")
    print("Risk model training completed successfully.")


if __name__ == "__main__":
    main()