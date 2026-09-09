import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

print("=" * 60)
print("TRAINING UNSAFE ACT MODEL (IMPROVED)")
print("=" * 60)

# Load improved balanced data
df = pd.read_csv("unsafe_act_balanced_enhanced.csv")
print(f"Loaded {len(df)} reports")


# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["report_text"].apply(clean_text)

# Encode labels
le = LabelEncoder()
df["risk_encoded"] = le.fit_transform(df["sif_binary"])
print(f"Labels: {le.classes_} (0=Non-SIF, 1=SIF)")

# Split data
X = df["clean_text"]
y = df["risk_encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training: {len(X_train)}, Testing: {len(X_test)}")

# Improved TF-IDF with more features
vectorizer = TfidfVectorizer(
    max_features=7000, min_df=2, max_df=0.85, ngram_range=(1, 3), stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Features: {X_train_tfidf.shape[1]}")

# Train with higher regularization to prevent overfitting
model = LogisticRegression(max_iter=2000, random_state=42, C=0.5)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred) * 100

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy: {accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-SIF", "SIF"]))

# Save models
joblib.dump(model, "unsafe_act_model_improved.pkl")
joblib.dump(vectorizer, "unsafe_act_vectorizer_improved.pkl")
joblib.dump(le, "unsafe_act_label_encoder_improved.pkl")

print("\n✅ Model saved successfully!")
