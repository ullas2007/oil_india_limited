import os
import re
import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

# ============================================================
# OIL GUARDIAN AI
# UNSAFE ACT - SIF PREDICTION MODEL V2
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "unsafe_act_sif_v2_cleaned.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "unsafe_act_sif_xgboost_v2.pkl")

METADATA_PATH = os.path.join(MODEL_DIR, "unsafe_act_sif_v2_metadata.pkl")

EMBEDDER_NAME = "all-MiniLM-L6-v2"


# ============================================================
# NORMALIZE REPORTS FOR GROUPING
# ============================================================


def normalize_report(text):
    """
    Creates a grouping key so very similar report templates
    remain in the same train/validation/test split.
    """

    text = str(text).lower()

    # Remove observation / incident / report numbers
    text = re.sub(r"\b(?:incident|observation|report|case|id)[\s:_-]*\d+\b", "", text)

    # Remove standalone numbers
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# START
# ============================================================

print("=" * 70)
print("OIL GUARDIAN AI - UNSAFE ACT SIF V2 TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Make sure the file is located at:\n"
        "data/processed/unsafe_act_sif_v2_cleaned.csv"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["report_text", "sif_binary"]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(f"Required column missing: {column}")


# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning dataset...")

df = df[["report_text", "sif_binary"]].copy()


# Convert text to normal Python strings
df["report_text"] = df["report_text"].astype(str).str.strip()


# Convert target to numeric
df["sif_binary"] = pd.to_numeric(df["sif_binary"], errors="coerce")


# Remove missing values
df = df.dropna(subset=["report_text", "sif_binary"])


# Remove empty reports
df = df[df["report_text"] != ""]


# Convert target to integer
df["sif_binary"] = df["sif_binary"].astype(int)


# Keep only valid binary labels
df = df[df["sif_binary"].isin([0, 1])]


# Remove exact duplicate reports
df = df.drop_duplicates(subset=["report_text"]).reset_index(drop=True)


print("\nCleaned dataset:")
print(f"Rows: {len(df)}")


# ============================================================
# SIF DISTRIBUTION
# ============================================================

print("\nSIF distribution:")

sif_distribution = df["sif_binary"].value_counts().sort_index()

print(sif_distribution.rename({0: "No SIF", 1: "SIF Potential"}))


# ============================================================
# CREATE GROUPS
# ============================================================

print("\nCreating report-template groups...")

df["group"] = df["report_text"].apply(normalize_report)

number_of_groups = df["group"].nunique()

print(f"Unique report groups: " f"{number_of_groups}")


# ============================================================
# PREPARE DATA
# ============================================================

X = df["report_text"].to_numpy()

y = df["sif_binary"].to_numpy()

groups = df["group"].to_numpy()


# ============================================================
# GROUPED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating grouped " "train/validation/test split...")


# ------------------------------------------------------------
# STEP 1: TEST SET
# ------------------------------------------------------------

gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)

train_val_idx, test_idx = next(gss_test.split(X, y, groups))


# ------------------------------------------------------------
# STEP 2: VALIDATION SET
# ------------------------------------------------------------

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)

train_idx_relative, val_idx_relative = next(
    gss_val.split(X[train_val_idx], y[train_val_idx], groups[train_val_idx])
)


train_idx = train_val_idx[train_idx_relative]

val_idx = train_val_idx[val_idx_relative]


# ============================================================
# CREATE DATA SPLITS
# ============================================================

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]


print("\nSplit sizes:")

print(f"Training:   {len(X_train)}")

print(f"Validation: {len(X_val)}")

print(f"Testing:    {len(X_test)}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(f"Train -> " f"No SIF: {(y_train == 0).sum()}, " f"SIF: {(y_train == 1).sum()}")

print(f"Validation -> " f"No SIF: {(y_val == 0).sum()}, " f"SIF: {(y_val == 1).sum()}")

print(f"Test -> " f"No SIF: {(y_test == 0).sum()}, " f"SIF: {(y_test == 1).sum()}")


# ============================================================
# VERIFY GROUP LEAKAGE
# ============================================================

print("\nChecking for group leakage...")


train_groups = set(groups[train_idx])

val_groups = set(groups[val_idx])

test_groups = set(groups[test_idx])


train_val_overlap = train_groups & val_groups

train_test_overlap = train_groups & test_groups

val_test_overlap = val_groups & test_groups


print(f"Train/Validation overlap: " f"{len(train_val_overlap)}")

print(f"Train/Test overlap: " f"{len(train_test_overlap)}")

print(f"Validation/Test overlap: " f"{len(val_test_overlap)}")


if train_val_overlap or train_test_overlap or val_test_overlap:

    raise RuntimeError("\nGROUP LEAKAGE DETECTED!\n" "Training stopped for safety.")


print("✅ No group leakage detected.")


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

print("\nLoading Sentence Transformer...")

embedder = SentenceTransformer(EMBEDDER_NAME)

print("✅ Sentence Transformer loaded.")


# ============================================================
# CREATE TRAINING EMBEDDINGS
# ============================================================

print("\nGenerating training embeddings...")

X_train_embeddings = embedder.encode(
    X_train.tolist(), batch_size=32, convert_to_numpy=True, show_progress_bar=True
)


# ============================================================
# CREATE VALIDATION EMBEDDINGS
# ============================================================

print("\nGenerating validation embeddings...")

X_val_embeddings = embedder.encode(
    X_val.tolist(), batch_size=32, convert_to_numpy=True, show_progress_bar=True
)


# ============================================================
# CREATE TEST EMBEDDINGS
# ============================================================

print("\nGenerating test embeddings...")

X_test_embeddings = embedder.encode(
    X_test.tolist(), batch_size=32, convert_to_numpy=True, show_progress_bar=True
)


# ============================================================
# EMBEDDING INFORMATION
# ============================================================

print("\nEmbedding shapes:")

print(f"Training:   " f"{X_train_embeddings.shape}")

print(f"Validation: " f"{X_val_embeddings.shape}")

print(f"Testing:    " f"{X_test_embeddings.shape}")


# ============================================================
# CALCULATE CLASS WEIGHT
# ============================================================

negative_count = np.sum(y_train == 0)

positive_count = np.sum(y_train == 1)


if positive_count == 0:
    raise ValueError("Training set contains no SIF=1 examples.")


scale_pos_weight = negative_count / positive_count


print(f"\nScale positive weight: " f"{scale_pos_weight:.3f}")


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\nTraining XGBoost V2...")


model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.04,
    max_depth=4,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)


model.fit(X_train_embeddings, y_train)


print("✅ XGBoost training completed.")


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)

print("VALIDATION RESULTS")

print("=" * 70)


val_probabilities = model.predict_proba(X_val_embeddings)[:, 1]


val_predictions = (val_probabilities >= 0.5).astype(int)


val_accuracy = accuracy_score(y_val, val_predictions)

val_precision = precision_score(y_val, val_predictions, zero_division=0)

val_recall = recall_score(y_val, val_predictions, zero_division=0)

val_f1 = f1_score(y_val, val_predictions, zero_division=0)

val_roc_auc = roc_auc_score(y_val, val_probabilities)


print(f"\nAccuracy:   " f"{val_accuracy:.4f}")

print(f"Precision:  " f"{val_precision:.4f}")

print(f"SIF Recall: " f"{val_recall:.4f}")

print(f"F1 Score:   " f"{val_f1:.4f}")

print(f"ROC-AUC:    " f"{val_roc_auc:.4f}")


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)

print("FINAL TEST RESULTS")

print("=" * 70)


test_probabilities = model.predict_proba(X_test_embeddings)[:, 1]


test_predictions = (test_probabilities >= 0.5).astype(int)


accuracy = accuracy_score(y_test, test_predictions)

precision = precision_score(y_test, test_predictions, zero_division=0)

recall = recall_score(y_test, test_predictions, zero_division=0)

f1 = f1_score(y_test, test_predictions, zero_division=0)

roc_auc = roc_auc_score(y_test, test_probabilities)


print(f"\nAccuracy:   " f"{accuracy:.4f}")

print(f"Precision:  " f"{precision:.4f}")

print(f"SIF Recall: " f"{recall:.4f}")

print(f"F1 Score:   " f"{f1:.4f}")

print(f"ROC-AUC:    " f"{roc_auc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["No SIF Potential", "SIF Potential"],
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, test_predictions)

print(cm)


print("\nMatrix format:")

print("[[True Negative, False Positive]")

print(" [False Negative, True Positive]]")


# ============================================================
# FALSE NEGATIVE ANALYSIS
# ============================================================

false_negative_mask = (y_test == 1) & (test_predictions == 0)


false_negative_count = int(false_negative_mask.sum())


print(f"\n⚠️ False Negatives: " f"{false_negative_count}")


if false_negative_count > 0:

    print("\nFalse-negative reports:")

    false_negative_reports = X_test[false_negative_mask]

    false_negative_probabilities = test_probabilities[false_negative_mask]

    for report, probability in zip(
        false_negative_reports, false_negative_probabilities
    ):

        print("\n----------------------------------------")

        print(f"SIF Probability: " f"{probability * 100:.2f}%")

        print(f"Report: {report}")

        print("----------------------------------------")


# ============================================================
# FALSE POSITIVE ANALYSIS
# ============================================================

false_positive_mask = (y_test == 0) & (test_predictions == 1)


false_positive_count = int(false_positive_mask.sum())


print(f"\n⚠️ False Positives: " f"{false_positive_count}")


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")


os.makedirs(MODEL_DIR, exist_ok=True)


joblib.dump(model, MODEL_PATH)


print(f"✅ Model saved to:\n" f"{MODEL_PATH}")


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model_version": "V2",
    "model_type": "Sentence Transformer + XGBoost",
    "embedding_model": EMBEDDER_NAME,
    "embedding_dimension": 384,
    "target": "sif_binary",
    "target_definition": {"0": "No SIF Potential", "1": "SIF Potential"},
    "decision_threshold": 0.50,
    "dataset_rows": len(df),
    "number_of_groups": number_of_groups,
    "train_size": len(X_train),
    "validation_size": len(X_val),
    "test_size": len(X_test),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "sif_recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "false_negatives": false_negative_count,
    "false_positives": false_positive_count,
    "confusion_matrix": cm.tolist(),
    "group_leakage_check": "passed",
}


joblib.dump(metadata, METADATA_PATH)


print(f"✅ Metadata saved to:\n" f"{METADATA_PATH}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("✅ UNSAFE ACT SIF V2 TRAINING COMPLETE")

print("=" * 70)


print("\nModel:")

print("Sentence Transformer → XGBoost")


print("\nNext step:")

print("Test the V2 model with real safety reports.")
