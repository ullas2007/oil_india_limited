#!/usr/bin/env python3
"""
train_unsafe_act_sif_v3.py

V3 training script for the OIL Guardian AI unsafe-act / SIF dataset.

Expected CSV columns:
    report_text
    category
    risk_level
    sif_label
    sif_binary
    facility
    department

What this script does:
1. Loads unsafe_act_v3_contrast_balanced.csv.
2. Cleans and validates the data.
3. Removes duplicate/empty reports.
4. Creates a stratified train/test split.
5. Trains a TF-IDF + Logistic Regression SIF classifier.
6. Uses both word and character n-grams to improve robustness to safety-report wording.
7. Evaluates accuracy, precision, recall, F1, confusion matrix and classification report.
8. Saves the trained model and metadata into models/.
9. Provides an optional --text argument for testing a new report.

Run from the project root:
    python src/train_unsafe_act_sif_v3.py

Or:
    python src/train_unsafe_act_sif_v3.py --csv data/unsafe_act_v3_contrast_balanced.csv

Test a new report:
    python src/train_unsafe_act_sif_v3.py --text "Worker entered a confined space without gas testing."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "unsafe_act_v3_contrast_balanced.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = DEFAULT_MODEL_DIR / "unsafe_act_sif_v3_model.joblib"
METADATA_FILE = DEFAULT_MODEL_DIR / "unsafe_act_sif_v3_metadata.json"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

REQUIRED_COLUMNS = [
    "report_text",
    "category",
    "risk_level",
    "sif_label",
    "sif_binary",
    "facility",
    "department",
]


# ---------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize safety-report text while keeping useful technical words."""
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Normalize common separators.
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Keep letters, numbers and useful safety symbols.
    text = re.sub(r"[^a-z0-9\s%/\-]", " ", text)

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{csv_path}\n\n"
            "Put unsafe_act_v3_contrast_balanced.csv in the project root "
            "or provide --csv with the correct path."
        )

    print(f"\nLoading dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing)
        )

    print(f"Original rows: {len(df)}")

    # Keep only rows with usable report text and a binary target.
    df = df.copy()
    df["report_text"] = df["report_text"].fillna("").map(clean_text)
    df["sif_binary"] = pd.to_numeric(df["sif_binary"], errors="coerce")

    before = len(df)
    df = df[df["report_text"].str.len() > 0]
    df = df[df["sif_binary"].isin([0, 1])]
    df = df.drop_duplicates(subset=["report_text"]).reset_index(drop=True)

    print(f"Rows after cleaning/deduplication: {len(df)}")
    print(f"Removed rows: {before - len(df)}")

    if len(df) < 20:
        raise ValueError("Dataset is too small after cleaning.")

    class_counts = df["sif_binary"].value_counts().sort_index()
    print("\nSIF binary distribution:")
    for label, count in class_counts.items():
        name = "Non-SIF" if int(label) == 0 else "SIF"
        print(f"  {int(label)} ({name}): {count}")

    if len(class_counts) < 2:
        raise ValueError("Both SIF and Non-SIF classes are required.")

    return df


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model() -> Pipeline:
    """
    Hybrid TF-IDF model:
      - word n-grams capture phrases such as 'confined space' and
        'lock out tag out'
      - character n-grams help with spelling variation and terminology
    """

    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=60000,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_model(model: Pipeline, x_test, y_test) -> dict:
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n" + "=" * 70)
    print("V3 MODEL EVALUATION")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Non-SIF", "SIF"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("Confusion matrix:")
    print("                 Pred Non-SIF   Pred SIF")
    print(f"Actual Non-SIF       {cm[0, 0]:<10} {cm[0, 1]}")
    print(f"Actual SIF           {cm[1, 0]:<10} {cm[1, 1]}")

    return {
        "accuracy": float(accuracy),
        "precision_sif": float(precision),
        "recall_sif": float(recall),
        "f1_sif": float(f1),
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(y_test)),
    }


# ---------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------

def predict_report(model: Pipeline, text: str) -> dict:
    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError("Prediction text is empty.")

    prediction = int(model.predict([cleaned])[0])

    # Logistic regression probability for class 1 (SIF).
    probabilities = model.predict_proba([cleaned])[0]
    classes = list(model.named_steps["classifier"].classes_)

    if 1 in classes:
        sif_probability = float(probabilities[classes.index(1)])
    else:
        sif_probability = 1.0 if prediction == 1 else 0.0

    label = "SIF" if prediction == 1 else "Non-SIF"

    return {
        "prediction": label,
        "sif_binary": prediction,
        "sif_probability": sif_probability,
        "confidence": max(sif_probability, 1.0 - sif_probability),
    }


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------

def train(csv_path: Path, model_dir: Path) -> tuple[Pipeline, dict]:
    df = load_dataset(csv_path)

    X = df["report_text"]
    y = df["sif_binary"].astype(int)

    # Stratification is important because SIF and Non-SIF must both be
    # represented in the train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDataset split:")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Testing samples : {len(x_test)}")

    print("\nBuilding V3 model...")
    model = build_model()

    print("Training...")
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)

    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    metadata = {
        "model_version": "V3",
        "task": "SIF binary classification",
        "target_column": "sif_binary",
        "label_mapping": {
            "0": "Non-SIF",
            "1": "SIF",
        },
        "dataset": csv_path.name,
        "total_clean_samples": int(len(df)),
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "feature_type": "word TF-IDF + character TF-IDF",
        "word_ngram_range": [1, 2],
        "char_ngram_range": [3, 5],
        "classifier": "LogisticRegression",
        "class_weight": "balanced",
        "metrics": metrics,
        "category_distribution": {
            str(k): int(v)
            for k, v in df["category"].value_counts().to_dict().items()
        },
        "risk_level_distribution": {
            str(k): int(v)
            for k, v in df["risk_level"].value_counts().to_dict().items()
        },
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)
    print(f"Model   : {MODEL_FILE}")
    print(f"Metadata: {METADATA_FILE}")

    return model, metadata


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the OIL Guardian AI V3 Unsafe Act SIF classifier."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help="Path to the V3 CSV dataset.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Directory where the trained model is saved.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional safety report to classify after training.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()

    try:
        model, _ = train(csv_path, model_dir)

        if args.text:
            result = predict_report(model, args.text)

            print("\n" + "=" * 70)
            print("NEW REPORT PREDICTION")
            print("=" * 70)
            print(f"Report     : {args.text}")
            print(f"Prediction : {result['prediction']}")
            print(f"SIF score  : {result['sif_probability']:.2%}")
            print(f"Confidence : {result['confidence']:.2%}")

    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    except Exception as exc:
        print(
            f"\nUnexpected error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
