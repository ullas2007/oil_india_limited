import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "data/raw/unsafe_act_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("========================================")
print("OIL GUARDIAN AI - UNSAFE ACT ML V2")
print("========================================")

print("\nDataset shape:", df.shape)


# ==========================================
# 2. SELECT REQUIRED COLUMNS
# ==========================================

df = df[["description", "risk_level"]].dropna()

# Convert labels to uppercase
df["risk_level"] = df["risk_level"].str.upper()

# Convert CRITICAL into HIGH
df["risk_level"] = df["risk_level"].replace({"CRITICAL": "HIGH"})

print("\nRisk distribution:")
print(df["risk_level"].value_counts())


# ==========================================
# 3. REMOVE CONFLICTING DUPLICATE DESCRIPTIONS
# ==========================================

# Find descriptions that have more than one risk label
label_counts = df.groupby("description")["risk_level"].nunique()

conflicting_descriptions = label_counts[label_counts > 1].index

print("\nConflicting descriptions:", len(conflicting_descriptions))

# Remove them completely
df = df[~df["description"].isin(conflicting_descriptions)].copy()

print("Dataset after removing conflicts:", df.shape)


# ==========================================
# 4. USE UNIQUE DESCRIPTIONS FOR EVALUATION
# ==========================================

X = df["description"]
y = df["risk_level"]

groups = df["description"]


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF features:", X_train_tfidf.shape[1])


# ==========================================
# 7. TRAIN LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)

model.fit(X_train_tfidf, y_train)


# ==========================================
# 8. EVALUATE MODEL
# ==========================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ==========================================
# 9. SAVE NEW MODEL
# ==========================================

MODEL_PATH = "models/unsafe_act_ml_v2.pkl"
VECTORIZER_PATH = "models/unsafe_act_vectorizer_v2.pkl"

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print("Model:", MODEL_PATH)
print("Vectorizer:", VECTORIZER_PATH)

print("\nTraining complete!")
