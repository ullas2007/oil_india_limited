#!/usr/bin/env python3

"""
Fair V2 vs V3 comparison for the OIL Guardian AI SIF classifier.

V2:
    Sentence Transformer: all-MiniLM-L6-v2
    Classifier: XGBoost
    Model: models/unsafe_act_sif_xgboost_v2.pkl

V3:
    Existing trained V3 model
    Model: models/unsafe_act_sif_v3_model.joblib

Both models are evaluated on the EXACT SAME test samples.

Run from the project root:

python compare_v2_v3.py --csv "data/raw/unsafe_act_v3_contrast_balanced.csv"
"""

from pathlib import Path
import argparse
import json
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# FILE LOCATIONS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

V2_MODEL_PATH = MODELS_DIR / "unsafe_act_sif_xgboost_v2.pkl"
V3_MODEL_PATH = MODELS_DIR / "unsafe_act_sif_v3_model.joblib"

V2_METADATA_PATH = MODELS_DIR / "unsafe_act_sif_v2_metadata.pkl"


# ============================================================
# LOAD V2 XGBOOST MODEL
# ============================================================

def load_v2_model():

    if not V2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nV2 model not found:\n{V2_MODEL_PATH}\n"
        )

    print(f"Loading V2 model:")
    print(f"  {V2_MODEL_PATH}")

    with open(V2_MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print(f"V2 model type: {type(model)}")

    return model


# ============================================================
# LOAD V3 MODEL
# ============================================================

def load_v3_model():

    if not V3_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nV3 model not found:\n{V3_MODEL_PATH}\n"
        )

    print(f"Loading V3 model:")
    print(f"  {V3_MODEL_PATH}")

    model = joblib.load(V3_MODEL_PATH)

    print(f"V3 model type: {type(model)}")

    return model


# ============================================================
# CREATE MINILM EMBEDDINGS
# ============================================================

def create_v2_embeddings(texts):

    print("\nLoading Sentence Transformer:")
    print("  all-MiniLM-L6-v2")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:

        raise ImportError(
            "\nThe sentence-transformers package is not installed.\n\n"
            "Install it using:\n"
            "pip install sentence-transformers\n"
        )

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating V2 embeddings...")

    embeddings = encoder.encode(
        texts.tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    print(f"Embedding shape: {embeddings.shape}")

    return embeddings


# ============================================================
# V2 PREDICTION
# ============================================================

def predict_v2(model, embeddings):

    print("\nGenerating V2 predictions...")

    predictions = model.predict(embeddings)

    predictions = np.asarray(predictions).astype(int)

    return predictions


# ============================================================
# V3 PREDICTION
# ============================================================

def predict_v3(model, texts):

    print("\nGenerating V3 predictions...")

    predictions = model.predict(texts)

    predictions = np.asarray(predictions).astype(int)

    return predictions


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(y_true, predictions):

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "confusion_matrix": cm,

        "predictions": predictions,
    }


# ============================================================
# PRINT CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(name, cm):

    print(f"\n{name} Confusion Matrix")

    print(
        "                 Pred Non-SIF   Pred SIF"
    )

    print(
        f"Actual Non-SIF       {cm[0, 0]:>5}       {cm[0, 1]:>5}"
    )

    print(
        f"Actual SIF           {cm[1, 0]:>5}       {cm[1, 1]:>5}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Fair comparison of V2 and V3 SIF classifiers."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the comparison dataset.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()

    print("=" * 72)
    print("OIL GUARDIAN AI")
    print("V2 vs V3 FAIR MODEL COMPARISON")
    print("=" * 72)

    print(f"\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print(f"\nDataset:")
    print(f"  {csv_path}")

    if not csv_path.exists():

        raise FileNotFoundError(
            f"\nCSV file not found:\n{csv_path}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading dataset...")

    df = pd.read_csv(csv_path)

    print(f"Original rows: {len(df)}")

    required_columns = [
        "report_text",
        "sif_binary",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # ========================================================
    # CLEAN DATA
    # ========================================================

    df["report_text"] = (
        df["report_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["sif_binary"] = pd.to_numeric(
        df["sif_binary"],
        errors="coerce",
    )

    df = df[
        (df["report_text"] != "")
        & (df["sif_binary"].isin([0, 1]))
    ].copy()

    df = df.drop_duplicates(
        subset=["report_text"]
    ).reset_index(drop=True)

    print(f"Rows after cleaning: {len(df)}")

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print("\nDataset label distribution:")

    counts = df["sif_binary"].value_counts().sort_index()

    print(
        f"  0 (Non-SIF): {counts.get(0, 0)}"
    )

    print(
        f"  1 (SIF):     {counts.get(1, 0)}"
    )

    # ========================================================
    # SAME TEST SPLIT FOR BOTH MODELS
    # ========================================================

    print("\nCreating identical stratified test split...")

    indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sif_binary"],
    )

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    X_test = test_df["report_text"]

    y_test = test_df["sif_binary"].astype(int).to_numpy()

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Testing samples : {len(test_df)}"
    )

    print(
        f"Test Non-SIF    : {(y_test == 0).sum()}"
    )

    print(
        f"Test SIF        : {(y_test == 1).sum()}"
    )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    print("\n" + "=" * 72)
    print("LOADING MODELS")
    print("=" * 72)

    v2_model = load_v2_model()

    v3_model = load_v3_model()

    # ========================================================
    # V2
    # ========================================================

    print("\n" + "=" * 72)
    print("V2 EVALUATION")
    print("=" * 72)

    v2_embeddings = create_v2_embeddings(
        X_test
    )

    v2_predictions = predict_v2(
        v2_model,
        v2_embeddings,
    )

    # Make sure predictions are binary.
    unique_v2 = np.unique(v2_predictions)

    print(
        f"V2 prediction classes: {unique_v2}"
    )

    if not set(unique_v2).issubset({0, 1}):

        raise ValueError(
            "V2 produced unexpected class labels: "
            f"{unique_v2}"
        )

    v2_results = evaluate_model(
        y_test,
        v2_predictions,
    )

    # ========================================================
    # V3
    # ========================================================

    print("\n" + "=" * 72)
    print("V3 EVALUATION")
    print("=" * 72)

    v3_predictions = predict_v3(
        v3_model,
        X_test,
    )

    unique_v3 = np.unique(v3_predictions)

    print(
        f"V3 prediction classes: {unique_v3}"
    )

    if not set(unique_v3).issubset({0, 1}):

        raise ValueError(
            "V3 produced unexpected class labels: "
            f"{unique_v3}"
        )

    v3_results = evaluate_model(
        y_test,
        v3_predictions,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 72)
    print("V2 vs V3 RESULTS")
    print("=" * 72)

    print()

    print(
        f"{'Metric':<15}"
        f"{'V2':>12}"
        f"{'V3':>12}"
        f"{'V3 - V2':>14}"
    )

    print("-" * 55)

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
    ]

    for display_name, key in metrics:

        v2_value = v2_results[key]

        v3_value = v3_results[key]

        difference = (
            v3_value - v2_value
        )

        print(
            f"{display_name:<15}"
            f"{v2_value:>12.4f}"
            f"{v3_value:>12.4f}"
            f"{difference:>+14.4f}"
        )

    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    print("\n" + "=" * 72)
    print("CONFUSION MATRICES")
    print("=" * 72)

    print_confusion_matrix(
        "V2",
        v2_results["confusion_matrix"],
    )

    print_confusion_matrix(
        "V3",
        v3_results["confusion_matrix"],
    )

    # ========================================================
    # CLASSIFICATION REPORTS
    # ========================================================

    print("\n" + "=" * 72)
    print("V2 CLASSIFICATION REPORT")
    print("=" * 72)

    print(
        classification_report(
            y_test,
            v2_predictions,
            labels=[0, 1],
            target_names=[
                "Non-SIF",
                "SIF",
            ],
            zero_division=0,
        )
    )

    print("\n" + "=" * 72)
    print("V3 CLASSIFICATION REPORT")
    print("=" * 72)

    print(
        classification_report(
            y_test,
            v3_predictions,
            labels=[0, 1],
            target_names=[
                "Non-SIF",
                "SIF",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # SAFETY ANALYSIS
    # ========================================================

    v2_cm = v2_results["confusion_matrix"]
    v3_cm = v3_results["confusion_matrix"]

    v2_false_negatives = int(v2_cm[1, 0])
    v3_false_negatives = int(v3_cm[1, 0])

    v2_false_positives = int(v2_cm[0, 1])
    v3_false_positives = int(v3_cm[0, 1])

    print("\n" + "=" * 72)
    print("SAFETY ANALYSIS")
    print("=" * 72)

    print(
        f"\nV2 SIF false negatives: "
        f"{v2_false_negatives}"
    )

    print(
        f"V3 SIF false negatives: "
        f"{v3_false_negatives}"
    )

    print(
        f"\nV2 false positives: "
        f"{v2_false_positives}"
    )

    print(
        f"V3 false positives: "
        f"{v3_false_positives}"
    )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    print("\n" + "=" * 72)
    print("FINAL COMPARISON VERDICT")
    print("=" * 72)

    v2_f1 = v2_results["f1"]
    v3_f1 = v3_results["f1"]

    if v3_f1 > v2_f1:

        print(
            "\nV3 has the higher F1-score."
        )

    elif v2_f1 > v3_f1:

        print(
            "\nV2 has the higher F1-score."
        )

    else:

        print(
            "\nV2 and V3 have the same F1-score."
        )

    print(
        f"\nV2 F1-score: {v2_f1:.4f}"
    )

    print(
        f"V3 F1-score: {v3_f1:.4f}"
    )

    if v3_results["recall"] > v2_results["recall"]:

        print(
            "\nV3 has better SIF recall."
        )

    elif v2_results["recall"] > v3_results["recall"]:

        print(
            "\nV2 has better SIF recall."
        )

    else:

        print(
            "\nBoth models have the same SIF recall."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_path = (
        MODELS_DIR /
        "v2_v3_comparison_results.json"
    )

    output = {

        "dataset": str(csv_path),

        "test_size": len(test_df),

        "random_state": RANDOM_STATE,

        "V2": {
            "model": str(V2_MODEL_PATH),
            "accuracy": float(
                v2_results["accuracy"]
            ),
            "precision": float(
                v2_results["precision"]
            ),
            "recall": float(
                v2_results["recall"]
            ),
            "f1": float(
                v2_results["f1"]
            ),
            "false_negatives": v2_false_negatives,
            "false_positives": v2_false_positives,
            "confusion_matrix":
                v2_results[
                    "confusion_matrix"
                ].tolist(),
        },

        "V3": {
            "model": str(V3_MODEL_PATH),
            "accuracy": float(
                v3_results["accuracy"]
            ),
            "precision": float(
                v3_results["precision"]
            ),
            "recall": float(
                v3_results["recall"]
            ),
            "f1": float(
                v3_results["f1"]
            ),
            "false_negatives": v3_false_negatives,
            "false_positives": v3_false_positives,
            "confusion_matrix":
                v3_results[
                    "confusion_matrix"
                ].tolist(),
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nResults saved to:\n{output_path}"
    )

    print("\n" + "=" * 72)
    print("COMPARISON COMPLETE")
    print("=" * 72)


if __name__ == "__main__":

    main()