import re
import joblib

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("models/unsafe_act_model.pkl")

vectorizer = joblib.load("models/unsafe_act_vectorizer.pkl")


# ============================================================
# TEXT CLEANING
# ============================================================


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PREDICT FUNCTION
# ============================================================


def predict_risk(report):

    cleaned = clean_text(report)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    probability_dict = dict(zip(model.classes_, probabilities))

    print("\n========================================")
    print("OIL GUARDIAN AI - UNSAFE ACT")
    print("========================================")

    print("\nReport:")
    print(report)

    print("\nPredicted Risk Level:")
    print(prediction)

    print("\nRisk Probabilities:")

    for risk in ["HIGH", "MEDIUM", "LOW"]:

        if risk in probability_dict:

            print(f"{risk}: " f"{probability_dict[risk] * 100:.2f}%")


# ============================================================
# TEST REPORTS
# ============================================================

predict_risk("""
    Worker bypassed safety lockout on high pressure line
    during maintenance. Lockout tagout procedure was not
    followed. Potential for catastrophic pressure release.
    """)


predict_risk("""
    Operator worked at height without fall protection.
    No safety harness was worn during elevated platform work.
    """)


predict_risk("""
    Worker correctly used fall protection equipment.
    Full body harness and lanyard were inspected before work.
    """)


predict_risk("""
    Worker used personal phone during a safety critical
    operation and was distracted from the task.
    """)


print("\n========================================")
print("TESTING COMPLETE")
print("========================================")
