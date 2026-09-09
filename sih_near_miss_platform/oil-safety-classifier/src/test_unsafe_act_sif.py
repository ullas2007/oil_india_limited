import os
import joblib
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "unsafe_act_sif_xgboost.pkl")

EMBEDDER_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading SIF model...")

model = joblib.load(MODEL_PATH)

print("Loading Sentence Transformer...")

embedder = SentenceTransformer(EMBEDDER_NAME)

print("\n✅ Model loaded successfully!")


# ============================================================
# MANUAL TESTING
# ============================================================

print("\n" + "=" * 60)
print("OIL GUARDIAN AI - UNSAFE ACT SIF TESTER")
print("=" * 60)

print("\nType a safety report to analyze it.")
print("Type 'exit' to stop.\n")


while True:

    report = input("Enter safety report:\n> ")

    if report.lower().strip() == "exit":
        print("\nExiting...")
        break

    if not report.strip():
        print("⚠️ Please enter a report.\n")
        continue

    # Create embedding
    embedding = embedder.encode([report], convert_to_numpy=True)

    # Predict probability
    probability = model.predict_proba(embedding)[0][1]

    # Convert to percentage
    sif_probability = probability * 100

    # Prediction
    prediction = 1 if probability >= 0.5 else 0

    print("\n" + "-" * 60)

    print(f"SIF Probability: {sif_probability:.2f}%")

    if prediction == 1:
        print("SIF Potential: YES")
    else:
        print("SIF Potential: NO")

    print("-" * 60 + "\n")
