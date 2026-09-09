import joblib
import numpy as np
import pandas as pd

from src.config import (
    ML_READY_FILE,
    EMBEDDINGS_FILE,
    SIF_MODEL_FILE,
    RISK_MODEL_FILE,
    RISK_LABEL_ENCODER_FILE,
    DASHBOARD_OUTPUT_FILE,
    SIF_THRESHOLD,
    create_project_directories
)


def get_priority(sif_probability, predicted_risk, confidence):
    if sif_probability >= 0.70 or predicted_risk == "Critical":
        return "Immediate Review"

    if sif_probability >= SIF_THRESHOLD or predicted_risk == "High":
        return "High Priority"

    if predicted_risk == "Medium" or confidence < 0.55:
        return "Manager Review"

    return "Routine Review"


def main():
    create_project_directories()

    required_files = [
        ML_READY_FILE,
        EMBEDDINGS_FILE,
        SIF_MODEL_FILE,
        RISK_MODEL_FILE,
        RISK_LABEL_ENCODER_FILE
    ]

    missing_files = [
        str(file_path)
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required file(s) missing:\n" + "\n".join(missing_files)
        )

    df = pd.read_csv(ML_READY_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Row mismatch: reports={len(df)}, embeddings={len(embeddings)}"
        )

    sif_model = joblib.load(SIF_MODEL_FILE)
    risk_model = joblib.load(RISK_MODEL_FILE)
    risk_encoder = joblib.load(RISK_LABEL_ENCODER_FILE)

    print(f"Reports loaded: {len(df)}")
    print("Generating SIF predictions...")

    sif_probabilities = sif_model.predict_proba(embeddings)[:, 1]
    predicted_sif = (sif_probabilities >= SIF_THRESHOLD).astype(int)

    print("Generating four-level risk predictions...")

    risk_probabilities = risk_model.predict_proba(embeddings)
    risk_class_ids = np.argmax(risk_probabilities, axis=1)
    predicted_risks = risk_encoder.inverse_transform(risk_class_ids)

    risk_confidence = risk_probabilities.max(axis=1)

    result = df.copy()

    result["predicted_sif_probability"] = np.round(
        sif_probabilities, 4
    )

    result["predicted_sif_potential"] = predicted_sif

    result["predicted_sif_label"] = np.where(
        predicted_sif == 1,
        "SIF Potential",
        "Non-SIF"
    )

    result["predicted_risk_level"] = predicted_risks

    result["risk_prediction_confidence"] = np.round(
        risk_confidence, 4
    )

    for class_index, class_name in enumerate(risk_encoder.classes_):
        column_name = (
            "risk_probability_"
            + class_name.lower().replace(" ", "_")
        )

        result[column_name] = np.round(
            risk_probabilities[:, class_index],
            4
        )

    result["review_priority"] = [
        get_priority(
            sif_probability=sif_probability,
            predicted_risk=predicted_risk,
            confidence=confidence
        )
        for sif_probability, predicted_risk, confidence in zip(
            sif_probabilities,
            predicted_risks,
            risk_confidence
        )
    ]

    result["requires_manager_review"] = np.where(
        result["review_priority"] == "Routine Review",
        "No",
        "Yes"
    )

    priority_order = {
        "Immediate Review": 1,
        "High Priority": 2,
        "Manager Review": 3,
        "Routine Review": 4
    }

    result["priority_rank"] = result["review_priority"].map(
        priority_order
    )

    result = result.sort_values(
        by=[
            "priority_rank",
            "predicted_sif_probability",
            "risk_prediction_confidence"
        ],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    result.to_csv(DASHBOARD_OUTPUT_FILE, index=False)

    print("\n--- Prediction Export Complete ---")
    print(f"Saved dashboard data: {DASHBOARD_OUTPUT_FILE}")

    print("\nPredicted SIF distribution:")
    print(result["predicted_sif_label"].value_counts())

    print("\nPredicted risk-level distribution:")
    print(result["predicted_risk_level"].value_counts())

    print("\nReview-priority distribution:")
    print(result["review_priority"].value_counts())

    print("\nFirst five dashboard rows:")
    preview_columns = [
        column for column in [
            "report_id",
            "location",
            "incident_type",
            "predicted_sif_label",
            "predicted_sif_probability",
            "predicted_risk_level",
            "risk_prediction_confidence",
            "review_priority"
        ]
        if column in result.columns
    ]

    print(result[preview_columns].head())


if __name__ == "__main__":
    main()