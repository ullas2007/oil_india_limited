import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import (
    EMBEDDINGS_FILE,
    ML_READY_FILE,
    MODELS_DIR,
    SIF_MODEL_FILE,
)


POSSIBLE_SIF_COLUMNS = [
    "binary_sif",
    "sif_potential",
    "sif_label",
    "sif",
    "is_sif",
    "final_sif_label",
    "final_binary_sif",
    "label",
]


def find_sif_column(df):
    normalized_columns = {
        column.strip().lower(): column
        for column in df.columns
    }

    for possible_name in POSSIBLE_SIF_COLUMNS:
        if possible_name in normalized_columns:
            return normalized_columns[possible_name]

    raise ValueError(
        "Could not find a SIF target column.\n"
        f"Available columns: {df.columns.tolist()}\n"
        f"Expected one of: {POSSIBLE_SIF_COLUMNS}"
    )


def convert_to_binary_sif(series):
    numeric_values = pd.to_numeric(series, errors="coerce")

    # Case 1: Dataset already has numeric 0 and 1 values
    if numeric_values.dropna().isin([0, 1]).all():
        return numeric_values

    # Case 2: Dataset uses text labels
    cleaned_values = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
    )

    label_map = {
        "1": 1,
        "yes": 1,
        "true": 1,
        "sif": 1,
        "sif potential": 1,
        "sif potential report": 1,
        "high potential": 1,
        "high potential sif": 1,

        "0": 0,
        "no": 0,
        "false": 0,
        "non sif": 0,
        "non sif potential": 0,
        "non sif report": 0,
        "non sif potential report": 0,
        "not sif": 0,
        "routine": 0,
    }

    return cleaned_values.map(label_map)


def main():
    print("Loading ML-ready data...")
    df = pd.read_csv(ML_READY_FILE)

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)

    print(f"Total reports: {len(df)}")
    print(f"Embedding matrix shape: {embeddings.shape}")

    if len(df) != len(embeddings):
        raise ValueError(
            f"Data mismatch: CSV has {len(df)} rows but embeddings "
            f"has {len(embeddings)} vectors."
        )

    print("\nAvailable columns:")
    print(df.columns.tolist())

    sif_column = find_sif_column(df)

    print(f"\nDetected SIF target column: {sif_column}")

    df["binary_sif"] = convert_to_binary_sif(df[sif_column])

    invalid_rows = df["binary_sif"].isna().sum()

    if invalid_rows > 0:
        print(f"\nIgnored rows with unrecognized SIF labels: {invalid_rows}")

        print("\nUnrecognized label values:")
        print(
            df.loc[
                df["binary_sif"].isna(),
                sif_column
            ]
            .astype(str)
            .value_counts()
            .head(20)
        )

    valid_rows = df["binary_sif"].isin([0, 1])

    df = df.loc[valid_rows].copy()
    embeddings = embeddings[valid_rows.to_numpy()]

    if len(df) == 0:
        raise ValueError(
            "No valid SIF labels remain after conversion. "
            "Check the printed label values and update label_map."
        )

    X = embeddings
    y = df["binary_sif"].astype(int).to_numpy()

    print("\nSIF label distribution after conversion:")
    print(pd.Series(y).value_counts().sort_index())

    if len(set(y)) < 2:
        raise ValueError(
            "Training needs both classes:\n"
            "0 = Non-SIF\n"
            "1 = SIF Potential"
        )

    class_counts = pd.Series(y).value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each SIF class needs at least 2 records for "
            "a stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Class imbalance weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost SIF model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).astype(int)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n--- SIF Model Evaluation ---")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Non-SIF", "SIF Potential"],
            zero_division=0,
        )
    )

    print("\nFirst 10 prediction probabilities:")
    preview_df = pd.DataFrame({
        "actual_sif": y_test[:10],
        "predicted_sif": predictions[:10],
        "sif_probability": probabilities[:10].round(4),
    })
    print(preview_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, SIF_MODEL_FILE)

    print(f"\nSaved SIF model to:")
    print(SIF_MODEL_FILE)

    print("\nSIF model training completed successfully.")


if __name__ == "__main__":
    main()