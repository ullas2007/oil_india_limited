#!/usr/bin/env python3

"""
OIL Guardian AI
V4 SIF Classifier

Leakage-resistant / generalization-focused training pipeline.

Expected CSV columns:
    report_text
    sif_binary

Labels:
    0 = Non-SIF
    1 = SIF

V4 improvements over V3:
    1. Removes explicit outcome/label phrases.
    2. Removes exact duplicate reports.
    3. Uses group-aware splitting.
    4. Uses separate train / validation / test data.
    5. Uses word + character TF-IDF features.
    6. Uses validation data for threshold selection.
    7. Keeps final test set untouched until final evaluation.
    8. Saves model + metadata.

IMPORTANT:
Do not use the final test set to tune the model.
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CSV = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "unsafe_act_v3_contrast_balanced.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "unsafe_act_sif_v4_model.joblib"
METADATA_PATH = MODEL_DIR / "unsafe_act_sif_v4_metadata.json"

RESULTS_PATH = MODEL_DIR / "unsafe_act_sif_v4_results.json"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Basic normalization without destroying safety terminology.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.lower()

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


# ============================================================
# REMOVE LABEL / OUTCOME LEAKAGE
# ============================================================

def remove_label_leakage(text):
    """
    Remove phrases that directly reveal the target label.

    Examples:
        "SIF potential"
        "SIF precursor"
        "non-SIF"
        "not a SIF"
        "safe work practice"

    The model should learn the actual safety situation,
    not explicit statements of the answer.
    """

    patterns = [

        # SIF terminology
        r"\bsif\s+potential\b",
        r"\bsif\s+precursor\b",
        r"\bsif\s+event\b",
        r"\bsif\s+case\b",
        r"\bsif\s+exposure\b",
        r"\bpotential\s+for\s+sif\b",
        r"\bserious\s+injury\s+and\s+fatality\b",
        r"\bserious\s+injury\s+or\s+fatality\b",

        # Explicit positive labels
        r"\bclassified\s+as\s+sif\b",
        r"\blabeled\s+as\s+sif\b",
        r"\bmarked\s+as\s+sif\b",
        r"\bidentified\s+as\s+sif\b",

        # Explicit negative labels
        r"\bnon[-\s]?sif\b",
        r"\bno\s+sif\s+potential\b",
        r"\bnot\s+a\s+sif\b",
        r"\bnot\s+sif\b",

        # Explicit outcome statements
        r"\bthis\s+is\s+a\s+sif\b",
        r"\bthis\s+was\s+a\s+sif\b",
        r"\bthis\s+creates\s+sif\s+potential\b",
        r"\bthis\s+created\s+sif\s+potential\b",

        # Generic synthetic-label phrases
        r"\bclassified\s+as\s+high\s+risk\b",
        r"\bclassified\s+as\s+low\s+risk\b",
        r"\bhigh[-\s]?risk\s+classification\b",
        r"\blow[-\s]?risk\s+classification\b",

        # Common synthetic explanation phrases
        r"\bthis\s+created\s+a\s+serious\s+risk\b",
        r"\bthis\s+was\s+a\s+safe\s+work\s+practice\b",
        r"\bthe\s+activity\s+was\s+safe\b",
        r"\bthe\s+activity\s+was\s+unsafe\b",
    ]

    cleaned = text

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


# ============================================================
# GROUP KEY
# ============================================================

def make_group_key(text):
    """
    Creates a conservative grouping key.

    Exact/near-template duplicates that differ only in numbers
    or simple identifiers are likely to receive the same group.

    This is intentionally conservative so that genuinely different
    safety reports are not unnecessarily merged.
    """

    key = text.lower()

    # Replace numbers
    key = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", key)

    # Replace common identifiers
    key = re.sub(
        r"\b(?:employee|worker|operator|technician|contractor)"
        r"\s*#?\s*[a-z0-9_-]+\b",
        "<person>",
        key,
    )

    # Normalize whitespace
    key = re.sub(r"\s+", " ", key).strip()

    return key


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(csv_path):

    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nCSV file not found:\n{csv_path}\n"
        )

    print("\nLoading dataset:")
    print(csv_path)

    df = pd.read_csv(csv_path)

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

    original_rows = len(df)

    # Basic cleaning
    df["report_text"] = (
        df["report_text"]
        .fillna("")
        .map(clean_text)
    )

    # Convert target
    df["sif_binary"] = pd.to_numeric(
        df["sif_binary"],
        errors="coerce",
    )

    # Remove invalid rows
    df = df[
        df["report_text"].str.len() > 0
    ]

    df = df[
        df["sif_binary"].isin([0, 1])
    ]

    # Convert label to int
    df["sif_binary"] = (
        df["sif_binary"]
        .astype(int)
    )

    # Remove exact duplicates BEFORE splitting
    df = (
        df
        .drop_duplicates(
            subset=["report_text"]
        )
        .reset_index(drop=True)
    )

    # Remove explicit label leakage
    df["clean_report"] = (
        df["report_text"]
        .map(remove_label_leakage)
    )

    # Remove reports that became empty
    df = df[
        df["clean_report"].str.len() > 0
    ]

    # Create grouping key
    df["group_key"] = (
        df["clean_report"]
        .map(make_group_key)
    )

    df = df.reset_index(drop=True)

    print("\nDataset cleaning:")
    print(f"Original rows          : {original_rows}")
    print(f"Rows after cleaning    : {len(df)}")
    print(
        f"Removed rows           : "
        f"{original_rows - len(df)}"
    )

    print("\nLabel distribution:")

    non_sif = int(
        (df["sif_binary"] == 0).sum()
    )

    sif = int(
        (df["sif_binary"] == 1).sum()
    )

    print(f"0 (Non-SIF): {non_sif}")
    print(f"1 (SIF)    : {sif}")

    print(
        f"\nUnique groups: "
        f"{df['group_key'].nunique()}"
    )

    return df


# ============================================================
# STRATIFIED GROUP SPLIT
# ============================================================

def create_group_splits(df):

    X = df["clean_report"].values
    y = df["sif_binary"].values
    groups = df["group_key"].values

    print("\nCreating group-aware splits...")

    # --------------------------------------------------------
    # First split:
    # 80% development
    # 20% final untouched test
    # --------------------------------------------------------

    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    outer_splits = list(
        outer.split(
            X,
            y,
            groups,
        )
    )

    # Select the fold whose test size is closest to 20%
    best_outer = min(
        outer_splits,
        key=lambda split: abs(
            len(split[1]) / len(df) - 0.20
        ),
    )

    development_idx, test_idx = best_outer

    # --------------------------------------------------------
    # Second split:
    # 75% development -> train
    # 25% development -> validation
    #
    # Overall:
    # train      ~= 60%
    # validation ~= 20%
    # test       ~= 20%
    # --------------------------------------------------------

    dev_df = df.iloc[development_idx]

    X_dev = dev_df["clean_report"].values
    y_dev = dev_df["sif_binary"].values
    groups_dev = dev_df["group_key"].values

    inner = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE + 1,
    )

    inner_splits = list(
        inner.split(
            X_dev,
            y_dev,
            groups_dev,
        )
    )

    best_inner = min(
        inner_splits,
        key=lambda split: abs(
            len(split[1]) / len(dev_df) - 0.25
        ),
    )

    train_dev_idx, val_dev_idx = best_inner

    train_idx = development_idx[
        train_dev_idx
    ]

    val_idx = development_idx[
        val_dev_idx
    ]

    print("\nFinal split:")

    print(
        f"Training samples   : {len(train_idx)}"
    )

    print(
        f"Validation samples : {len(val_idx)}"
    )

    print(
        f"Testing samples    : {len(test_idx)}"
    )

    print(
        f"\nTraining SIF       : "
        f"{int(df.iloc[train_idx]['sif_binary'].sum())}"
    )

    print(
        f"Validation SIF     : "
        f"{int(df.iloc[val_idx]['sif_binary'].sum())}"
    )

    print(
        f"Test SIF           : "
        f"{int(df.iloc[test_idx]['sif_binary'].sum())}"
    )

    # --------------------------------------------------------
    # Verify group separation
    # --------------------------------------------------------

    train_groups = set(
        df.iloc[train_idx]["group_key"]
    )

    val_groups = set(
        df.iloc[val_idx]["group_key"]
    )

    test_groups = set(
        df.iloc[test_idx]["group_key"]
    )

    train_val_overlap = (
        train_groups & val_groups
    )

    train_test_overlap = (
        train_groups & test_groups
    )

    val_test_overlap = (
        val_groups & test_groups
    )

    print("\nGroup leakage check:")

    print(
        f"Train/Validation overlap: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train/Test overlap      : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation/Test overlap : "
        f"{len(val_test_overlap)}"
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):
        raise RuntimeError(
            "Group leakage detected."
        )

    print("Group leakage check: PASSED")

    return (
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print("\nBuilding V4 model...")

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=30000,
    )

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.99,
        sublinear_tf=True,
        max_features=40000,
    )

    features = FeatureUnion(
        [
            (
                "word",
                word_vectorizer,
            ),
            (
                "char",
                char_vectorizer,
            ),
        ]
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        C=2.0,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return model


# ============================================================
# THRESHOLD METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


# ============================================================
# SELECT THRESHOLD
# ============================================================

def select_threshold(
    y_validation,
    validation_probabilities,
):

    print("\nSelecting decision threshold...")
    print(
        "Threshold selection uses VALIDATION data only."
    )

    thresholds = np.arange(
        0.30,
        0.81,
        0.01,
    )

    results = []

    for threshold in thresholds:

        result = calculate_metrics(
            y_validation,
            validation_probabilities,
            threshold,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Safety-oriented rule:
    #
    # Prefer thresholds with zero FN when possible.
    # Among those, maximize F1.
    #
    # If no threshold achieves zero FN,
    # maximize F1 while strongly penalizing FN.
    # --------------------------------------------------------

    zero_fn = results_df[
        results_df["false_negatives"] == 0
    ]

    if len(zero_fn) > 0:

        selected = zero_fn.sort_values(
            [
                "f1",
                "precision",
            ],
            ascending=False,
        ).iloc[0]

        reason = (
            "Selected from validation thresholds "
            "with zero SIF false negatives; "
            "highest F1."
        )

    else:

        results_df["safety_score"] = (
            results_df["f1"]
            - 0.05
            * results_df["false_negatives"]
        )

        selected = results_df.sort_values(
            [
                "safety_score",
                "f1",
            ],
            ascending=False,
        ).iloc[0]

        reason = (
            "No zero-FN threshold was available; "
            "selected using safety-weighted F1."
        )

    selected_threshold = float(
        selected["threshold"]
    )

    print(
        f"\nSelected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Reason: {reason}"
    )

    print("\nValidation threshold summary:")

    display_columns = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
    ]

    # Show every 0.05 threshold
    display_df = results_df[
        results_df["threshold"].apply(
            lambda x: abs(
                (x * 100) % 5
            ) < 0.001
        )
    ]

    print(
        display_df[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
            },
        )
    )

    return (
        selected_threshold,
        results_df,
        reason,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    threshold,
    name,
):

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    metrics = calculate_metrics(
        y,
        probabilities,
        threshold,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print("\n" + "=" * 72)
    print(f"{name.upper()} EVALUATION")
    print("=" * 72)

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"SIF Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score  : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"\nFalse Positives: "
        f"{metrics['false_positives']}"
    )

    print(
        f"False Negatives: "
        f"{metrics['false_negatives']}"
    )

    print("\nConfusion Matrix:")
    print(
        "                 Pred Non-SIF   Pred SIF"
    )

    print(
        f"Actual Non-SIF   "
        f"{metrics['true_negatives']:>10}   "
        f"{metrics['false_positives']:>8}"
    )

    print(
        f"Actual SIF       "
        f"{metrics['false_negatives']:>10}   "
        f"{metrics['true_positives']:>8}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Non-SIF",
                "SIF",
            ],
            digits=4,
            zero_division=0,
        )
    )

    return (
        metrics,
        probabilities,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train leakage-resistant OIL Guardian AI V4."
        )
    )

    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to training CSV.",
    )

    args = parser.parse_args()

    csv_path = (
        Path(args.csv)
        .expanduser()
        .resolve()
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("OIL GUARDIAN AI")
    print("V4 LEAKAGE-RESISTANT SIF CLASSIFIER")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_dataset(
        csv_path
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train_idx,
        val_idx,
        test_idx,
    ) = create_group_splits(df)

    X_train = df.iloc[
        train_idx
    ]["clean_report"].values

    y_train = df.iloc[
        train_idx
    ]["sif_binary"].values

    X_val = df.iloc[
        val_idx
    ]["clean_report"].values

    y_val = df.iloc[
        val_idx
    ]["sif_binary"].values

    X_test = df.iloc[
        test_idx
    ]["clean_report"].values

    y_test = df.iloc[
        test_idx
    ]["sif_binary"].values

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model()

    print("\nTraining V4...")
    print(
        "This may take a few minutes."
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------------
    # VALIDATION PROBABILITIES
    # --------------------------------------------------------

    print(
        "\nGenerating validation probabilities..."
    )

    validation_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    (
        selected_threshold,
        threshold_results,
        threshold_reason,
    ) = select_threshold(
        y_val,
        validation_probabilities,
    )

    # --------------------------------------------------------
    # VALIDATION EVALUATION
    # --------------------------------------------------------

    validation_metrics, _, _ = evaluate_model(
        model,
        X_val,
        y_val,
        selected_threshold,
        "V4 VALIDATION",
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print(
        "\nIMPORTANT:"
    )

    print(
        "The final test set has NOT been used "
        "for threshold selection."
    )

    print(
        "\nEvaluating on untouched test set..."
    )

    test_metrics, test_probabilities, test_predictions = (
        evaluate_model(
            model,
            X_test,
            y_test,
            selected_threshold,
            "V4 FINAL TEST",
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("SAVING V4 MODEL")
    print("=" * 72)

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"Model saved:\n{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {

        "model_version": "V4",

        "model_type": (
            "Word + Character TF-IDF "
            "with balanced Logistic Regression"
        ),

        "target": "sif_binary",

        "target_definition": {
            "0": "Non-SIF",
            "1": "SIF Potential",
        },

        "dataset": str(csv_path),

        "dataset_rows_after_cleaning": int(
            len(df)
        ),

        "unique_groups": int(
            df["group_key"].nunique()
        ),

        "train_size": int(
            len(train_idx)
        ),

        "validation_size": int(
            len(val_idx)
        ),

        "test_size": int(
            len(test_idx)
        ),

        "random_state": RANDOM_STATE,

        "group_leakage_check": "passed",

        "label_leakage_removal": True,

        "threshold_selection": (
            "validation_only"
        ),

        "decision_threshold": (
            selected_threshold
        ),

        "threshold_selection_reason": (
            threshold_reason
        ),

        "validation_metrics": (
            validation_metrics
        ),

        "final_test_metrics": (
            test_metrics
        ),

        "feature_configuration": {
            "word_ngram_range": [1, 2],
            "word_max_features": 30000,
            "char_ngram_range": [3, 5],
            "char_max_features": 40000,
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
        },

        "v3_benchmark_reference": {
            "accuracy": 0.9913,
            "precision": 0.9831,
            "sif_recall": 1.0,
            "f1": 0.9915,
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = {

        "model_version": "V4",

        "decision_threshold": (
            selected_threshold
        ),

        "validation_metrics": (
            validation_metrics
        ),

        "final_test_metrics": (
            test_metrics
        ),

        "threshold_analysis": (
            threshold_results.to_dict(
                orient="records"
            )
        ),

    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("V4 TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"\nFinal threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        "\nFINAL TEST RESULTS"
    )

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"SIF Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"\nFalse Positives: "
        f"{test_metrics['false_positives']}"
    )

    print(
        f"False Negatives: "
        f"{test_metrics['false_negatives']}"
    )

    print("\nFiles created:")

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print(
        f"\nMetadata:\n{METADATA_PATH}"
    )

    print(
        f"\nResults:\n{RESULTS_PATH}"
    )

    print("\n" + "=" * 72)
    print("DO NOT DELETE V2 OR V3")
    print("=" * 72)

    print(
        "\nV2 and V3 remain available as baseline models."
    )

    print(
        "\nV4 is now ready for independent unseen-data testing."
    )


if __name__ == "__main__":
    main()