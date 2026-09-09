import os
import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "unsafe_act_dataset_with_sif.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "unsafe_act_sif_xgboost.pkl")

EMBEDDER_NAME = "all-MiniLM-L6-v2"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("OIL GUARDIAN AI")
print("Unsafe Act SIF Potential Model")
print("=" * 60)

print("\n📊 Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["report_text", "sif_binary"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column '{column}' was not found in the dataset.")


# ============================================================
# 3. CLEAN DATA
# ============================================================

df = df[["report_text", "sif_binary"]].copy()

df["report_text"] = df["report_text"].fillna("").astype(str).str.strip()

df = df[df["report_text"] != ""]

df["sif_binary"] = pd.to_numeric(df["sif_binary"], errors="coerce")

df = df.dropna(subset=["sif_binary"])

df["sif_binary"] = df["sif_binary"].astype(int)


# Make sure only 0 and 1 exist
invalid_labels = sorted(set(df["sif_binary"].unique()) - {0, 1})

if invalid_labels:
    raise ValueError(f"Invalid sif_binary values found: {invalid_labels}")


print("\n✅ Clean dataset shape:", df.shape)

print("\nSIF distribution:")
print(df["sif_binary"].value_counts().sort_index())

print("\nMeaning:")
print("0 = No SIF potential")
print("1 = SIF potential")


# ============================================================
# 4. REMOVE DUPLICATE REPORTS
# ============================================================

before = len(df)

df = df.drop_duplicates(subset=["report_text"]).reset_index(drop=True)

removed = before - len(df)

print(f"\n🧹 Duplicate reports removed: {removed}")
print(f"Final dataset size: {len(df)}")


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_text = df["report_text"]
y = df["sif_binary"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.20, random_state=42, stratify=y
)

print("\n📚 Training samples:", len(X_train_text))
print("🧪 Testing samples:", len(X_test_text))


# ============================================================
# 6. LOAD SENTENCE TRANSFORMER
# ============================================================

print("\n🧠 Loading Sentence Transformer...")
print(f"Model: {EMBEDDER_NAME}")

embedder = SentenceTransformer(EMBEDDER_NAME)

print("✅ Embedding model loaded.")


# ============================================================
# 7. CREATE EMBEDDINGS
# ============================================================

print("\n🔄 Creating training embeddings...")

X_train = embedder.encode(
    X_train_text.tolist(), show_progress_bar=True, convert_to_numpy=True
)

print("\n🔄 Creating testing embeddings...")

X_test = embedder.encode(
    X_test_text.tolist(), show_progress_bar=True, convert_to_numpy=True
)

print("\nEmbedding shape:")
print("Training:", X_train.shape)
print("Testing :", X_test.shape)


# ============================================================
# 8. TRAIN XGBOOST
# ============================================================

print("\n🤖 Training XGBoost SIF classifier...")

# Calculate class imbalance weight
negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

scale_pos_weight = negative_count / positive_count

print(f"Non-SIF training samples: {negative_count}")
print(f"SIF training samples:     {positive_count}")
print(f"Scale positive weight:    {scale_pos_weight:.3f}")


model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("✅ XGBoost training completed.")


# ============================================================
# 9. EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nROC-AUC:")
print(f"{roc_auc_score(y_test, probabilities):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No SIF Potential", "SIF Potential"],
        digits=4,
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_PATH)

print("\n💾 Model saved:")
print(MODEL_PATH)


# ============================================================
# 11. SAVE EMBEDDING INFORMATION
# ============================================================

metadata = {
    "embedding_model": EMBEDDER_NAME,
    "embedding_dimension": int(X_train.shape[1]),
    "target": "sif_binary",
    "class_0": "No SIF potential",
    "class_1": "SIF potential",
}

metadata_path = os.path.join(MODEL_DIR, "unsafe_act_sif_metadata.pkl")

joblib.dump(metadata, metadata_path)

print("\n💾 Metadata saved:")
print(metadata_path)


print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE")
print("=" * 60)
