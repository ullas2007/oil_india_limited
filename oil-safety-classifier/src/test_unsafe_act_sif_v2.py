import os
import joblib
from sentence_transformers import SentenceTransformer

# ============================================================
# OIL GUARDIAN AI - UNSAFE ACT SIF V2 TESTER
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "unsafe_act_sif_xgboost_v2.pkl")

EMBEDDER_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("OIL GUARDIAN AI - UNSAFE ACT SIF V2 TESTER")
print("=" * 70)

print("\nLoading SIF V2 model...")

if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    print(MODEL_PATH)
    exit()

model = joblib.load(MODEL_PATH)

print("Loading Sentence Transformer...")
embedder = SentenceTransformer(EMBEDDER_NAME)

print("\n✅ Model loaded successfully!")


# ============================================================
# TEST LOOP
# ============================================================

print("\n" + "=" * 70)
print("ENTER SAFETY REPORT")
print("=" * 70)

print("\nType a safety report below.")
print("Type 'exit' to stop.\n")


while True:

    report = input("Report:\n> ")

    # Exit
    if report.lower().strip() == "exit":
        print("\nExiting OIL Guardian AI...")
        break

    # Empty input
    if not report.strip():
        print("⚠️ Please enter a safety report.\n")
        continue

    # ========================================================
    # GENERATE EMBEDDING
    # ========================================================

    embedding = embedder.encode([report], convert_to_numpy=True)

    # ========================================================
    # PREDICTION
    # ========================================================

    probability = model.predict_proba(embedding)[0][1]

    sif_probability = probability * 100

    prediction = 1 if probability >= 0.5 else 0

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "-" * 70)

    print(f"SIF Probability: {sif_probability:.2f}%")

    if prediction == 1:

        print("🚨 SIF Potential: YES")
        print("⚠️ HIGH-PRIORITY SAFETY REPORT")

    else:

        print("✅ SIF Potential: NO")
        print("ℹ️ No significant SIF potential detected")

    print("-" * 70)
    print()
