import joblib

MODEL_PATH = "models/unsafe_act_sif_v3_model.joblib"

print("Loading V3 model...")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print("ERROR loading model:")
    print(e)
    input("\nPress Enter to exit...")
    raise SystemExit

print("OIL Guardian AI - V3 SIF Classifier")
print("=" * 50)

while True:

    report = input("\nEnter safety report (or type 'exit'): ").strip()

    if report.lower() == "exit":
        print("Exiting...")
        break

    if not report:
        print("Please enter a report.")
        continue

    try:
        prediction = int(model.predict([report])[0])

        print("\nPrediction:")

        if prediction == 1:
            print("SIF POTENTIAL")
            print("Risk classification: 1")
        else:
            print("NON-SIF")
            print("Risk classification: 0")

        try:
            probabilities = model.predict_proba([report])[0]

            print(
                f"Non-SIF probability: {probabilities[0] * 100:.2f}%"
            )

            print(
                f"SIF probability:     {probabilities[1] * 100:.2f}%"
            )

        except Exception:
            pass

    except Exception as e:
        print("\nERROR during prediction:")
        print(e)