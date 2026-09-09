import re
import pandas as pd

from src.config import (
    RAW_DATA_FILE,
    STANDARDIZED_FILE,
    ML_READY_FILE,
    create_project_directories
)


def clean_text(text):
    text = str(text).strip()

    text = re.sub(r"\s+", " ", text)

    replacements = {
        r"\bloto\b": "lockout tagout",
        r"\bptw\b": "permit to work",
        r"\blel\b": "lower explosive limit",
        r"\bh2s\b": "hydrogen sulfide",
        r"\bhc\b": "hydrocarbon",
        r"\bscba\b": "self contained breathing apparatus"
    }

    for pattern, replacement in replacements.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text.strip()


def build_model_text(row):
    return (
        f"[REPORT_TYPE] {row['report_type']} "
        f"[SITE] {row['site']} "
        f"[ACTIVITY] {row['activity']} "
        f"[BARRIER_FAILURE] {row['barrier_failure']} "
        f"[IOGP_RULE] {row['iogp_life_saving_rule']} "
        f"[TEXT] {row['description_clean']}"
    )


def main():
    create_project_directories()

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset was not found: {RAW_DATA_FILE}"
        )

    df = pd.read_csv(RAW_DATA_FILE)

    df = df.rename(
        columns={
            "source_type": "report_type",
            "report_text": "description_raw",
            "location_type": "site"
        }
    )

    required_columns = [
        "report_id",
        "report_type",
        "description_raw",
        "site",
        "activity",
        "barrier_failure",
        "iogp_life_saving_rule",
        "risk_level",
        "sif_potential"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    text_columns = [
        "report_id",
        "report_type",
        "description_raw",
        "site",
        "activity",
        "barrier_failure",
        "iogp_life_saving_rule"
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    df["risk_level"] = (
        df["risk_level"]
        .fillna("Medium")
        .astype(str)
        .str.strip()
        .str.title()
    )

    allowed_risk_levels = {
        "Low",
        "Medium",
        "High",
        "Critical"
    }

    invalid_risk = ~df["risk_level"].isin(
        allowed_risk_levels
    )

    if invalid_risk.any():
        invalid_values = (
            df.loc[invalid_risk, "risk_level"]
            .astype(str)
            .unique()
        )

        raise ValueError(
            "Invalid risk-level values found: "
            + ", ".join(invalid_values)
        )

    df["sif_potential"] = pd.to_numeric(
        df["sif_potential"],
        errors="coerce"
    )

    if not df["sif_potential"].isin([0, 1]).all():
        raise ValueError(
            "Column 'sif_potential' must contain only 0 or 1."
        )

    df["sif_potential"] = (
        df["sif_potential"]
        .astype(int)
    )

    df["description_clean"] = (
        df["description_raw"]
        .apply(clean_text)
    )

    df["model_text"] = df.apply(
        build_model_text,
        axis=1
    )

    standardized_columns = [
        "report_id",
        "report_type",
        "description_raw",
        "description_clean",
        "site",
        "activity",
        "barrier_failure",
        "iogp_life_saving_rule",
        "risk_level",
        "sif_potential",
        "case_basis"
    ]

    standardized_columns = [
        column
        for column in standardized_columns
        if column in df.columns
    ]

    df[standardized_columns].to_csv(
        STANDARDIZED_FILE,
        index=False
    )

    ml_columns = [
        "report_id",
        "report_type",
        "site",
        "activity",
        "barrier_failure",
        "iogp_life_saving_rule",
        "description_raw",
        "description_clean",
        "model_text",
        "risk_level",
        "sif_potential"
    ]

    df[ml_columns].to_csv(
        ML_READY_FILE,
        index=False
    )

    print("--- Data Preparation Complete ---")
    print(f"Total reports prepared: {len(df)}")

    print("\nRisk-level distribution:")
    print(df["risk_level"].value_counts())

    print("\nSIF-potential distribution:")
    print(df["sif_potential"].value_counts().sort_index())

    print(f"\nSaved: {STANDARDIZED_FILE}")
    print(f"Saved: {ML_READY_FILE}")


if __name__ == "__main__":
    main()