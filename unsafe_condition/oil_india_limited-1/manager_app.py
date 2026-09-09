from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    BASE_DIR,
    DASHBOARD_OUTPUT_FILE,
    SUBMISSIONS_CSV,
    create_project_directories,
)


create_project_directories()


st.set_page_config(
    page_title="Manager Near-Miss Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_submissions():
    if SUBMISSIONS_CSV.exists():
        return pd.read_csv(SUBMISSIONS_CSV)

    return pd.DataFrame()


def save_submissions(df: pd.DataFrame):
    df.to_csv(SUBMISSIONS_CSV, index=False)


def safe_value(row, column_name, default="Not available"):
    if column_name not in row.index:
        return default

    value = row[column_name]

    if pd.isna(value):
        return default

    value = str(value).strip()

    return value if value else default


st.title("📊 Manager Near-Miss Dashboard")
st.caption(
    "Review employee near-miss reports, attachments, and AI-generated risk analytics."
)

page = st.sidebar.radio(
    "Dashboard Navigation",
    [
        "Employee Submissions",
        "AI Risk Dashboard",
    ],
)


if page == "Employee Submissions":
    st.header("Employee Submissions")

    submissions_df = load_submissions()

    if submissions_df.empty:
        st.warning(
            "No employee submissions are available yet. "
            "Submit a report from employee_app.py first."
        )
        st.stop()

    # Make older CSV files compatible with review fields
    if "manager_review" not in submissions_df.columns:
        submissions_df["manager_review"] = "Pending"

    if "manager_comment" not in submissions_df.columns:
        submissions_df["manager_comment"] = ""

    if "status" not in submissions_df.columns:
        submissions_df["status"] = "Submitted"

    total_reports = len(submissions_df)

    pending_reports = len(
        submissions_df[
            submissions_df["manager_review"].fillna("Pending") == "Pending"
        ]
    )

    reviewed_reports = len(
        submissions_df[
            submissions_df["manager_review"].fillna("") == "Reviewed"
        ]
    )

    critical_review_reports = len(
        submissions_df[
            submissions_df["manager_review"].fillna("") == "Immediate Review"
        ]
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total Employee Reports", total_reports)
    m2.metric("Pending Review", pending_reports)
    m3.metric("Reviewed", reviewed_reports)
    m4.metric("Immediate Review", critical_review_reports)

    st.divider()

    st.subheader("Filter Employee Reports")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        review_filter = st.selectbox(
            "Filter by manager review",
            [
                "All",
                "Pending",
                "Reviewed",
                "Immediate Review",
                "Closed",
            ],
        )

    with filter_col2:
        sites = ["All"] + sorted(
            submissions_df["site"]
            .fillna("Unknown")
            .astype(str)
            .unique()
            .tolist()
        )

        site_filter = st.selectbox(
            "Filter by site",
            sites,
        )

    with filter_col3:
        report_types = ["All"] + sorted(
            submissions_df["report_type"]
            .fillna("Unknown")
            .astype(str)
            .unique()
            .tolist()
        )

        report_type_filter = st.selectbox(
            "Filter by report type",
            report_types,
        )

    filtered_df = submissions_df.copy()

    if review_filter != "All":
        filtered_df = filtered_df[
            filtered_df["manager_review"].fillna("Pending") == review_filter
        ]

    if site_filter != "All":
        filtered_df = filtered_df[
            filtered_df["site"].fillna("Unknown").astype(str) == site_filter
        ]

    if report_type_filter != "All":
        filtered_df = filtered_df[
            filtered_df["report_type"].fillna("Unknown").astype(str)
            == report_type_filter
        ]

    st.subheader("All Employee Queries / Reports")

    display_columns = [
        "report_id",
        "submitted_at",
        "employee_name",
        "report_type",
        "site",
        "unit",
        "activity",
        "status",
        "manager_review",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]

    if "submitted_at" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            "submitted_at",
            ascending=False,
        )

    st.dataframe(
        filtered_df[available_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Employee Reports CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="employee_near_miss_reports.csv",
        mime="text/csv",
    )

    if filtered_df.empty:
        st.info("No reports match the selected filters.")
        st.stop()

    st.divider()
    st.subheader("Open and Review One Report")

    selected_report_id = st.selectbox(
        "Select report ID",
        filtered_df["report_id"].astype(str).tolist(),
    )

    selected_index = submissions_df.index[
        submissions_df["report_id"].astype(str) == selected_report_id
    ][0]

    selected_row = submissions_df.loc[selected_index]

    overview_col1, overview_col2 = st.columns(2)

    with overview_col1:
        st.write("**Report ID:**", safe_value(selected_row, "report_id"))
        st.write("**Submitted at:**", safe_value(selected_row, "submitted_at"))
        st.write("**Employee:**", safe_value(selected_row, "employee_name"))
        st.write("**Employee ID:**", safe_value(selected_row, "employee_id"))
        st.write("**Report type:**", safe_value(selected_row, "report_type"))
        st.write("**Reporter type:**", safe_value(selected_row, "reporter_type"))

    with overview_col2:
        st.write("**Site:**", safe_value(selected_row, "site"))
        st.write("**Unit:**", safe_value(selected_row, "unit"))
        st.write("**Activity:**", safe_value(selected_row, "activity"))
        st.write("**Equipment / area:**", safe_value(selected_row, "equipment"))
        st.write("**Language:**", safe_value(selected_row, "language"))
        st.write("**Current review state:**", safe_value(selected_row, "manager_review"))

    st.subheader("Employee Description")
    st.info(safe_value(selected_row, "description"))

    st.subheader("Immediate Action Taken")
    st.write(safe_value(selected_row, "immediate_action"))

    st.subheader("Attachments")

    attachment_col1, attachment_col2 = st.columns(2)

    with attachment_col1:
        image_path_text = safe_value(selected_row, "image_path", default="")

        if image_path_text:
            image_path = BASE_DIR / image_path_text

            if image_path.exists():
                st.image(
                    str(image_path),
                    caption="Employee-uploaded safety image",
                    use_container_width=True,
                )
            else:
                st.warning("Image path is saved, but the image file was not found.")
        else:
            st.info("No image was uploaded.")

    with attachment_col2:
        audio_path_text = safe_value(selected_row, "audio_path", default="")

        if audio_path_text:
            audio_path = BASE_DIR / audio_path_text

            if audio_path.exists():
                st.audio(str(audio_path))
            else:
                st.warning("Audio path is saved, but the audio file was not found.")
        else:
            st.info("No voice recording was uploaded.")

    st.divider()
    st.subheader("Manager Review")

    current_review = safe_value(
        selected_row,
        "manager_review",
        default="Pending",
    )

    review_options = [
        "Pending",
        "Reviewed",
        "Immediate Review",
        "Closed",
    ]

    if current_review not in review_options:
        current_review = "Pending"

    with st.form(f"manager_review_form_{selected_report_id}"):
        manager_review = st.selectbox(
            "Review status",
            review_options,
            index=review_options.index(current_review),
        )

        manager_comment = st.text_area(
            "Manager / HSE comment",
            value=safe_value(selected_row, "manager_comment", default=""),
            placeholder=(
                "Example: HSE team assigned. Inspect area before next shift."
            ),
            height=120,
        )

        report_status = st.selectbox(
            "Operational status",
            [
                "Submitted",
                "Under Review",
                "Action Assigned",
                "Closed",
            ],
            index=0,
        )

        save_review_button = st.form_submit_button(
            "Save Manager Review",
            type="primary",
        )

    if save_review_button:
        submissions_df.loc[selected_index, "manager_review"] = manager_review
        submissions_df.loc[selected_index, "manager_comment"] = manager_comment.strip()
        submissions_df.loc[selected_index, "status"] = report_status

        save_submissions(submissions_df)

        st.success(
            f"Manager review saved successfully for report {selected_report_id}."
        )
        st.rerun()


elif page == "AI Risk Dashboard":
    st.header("AI Risk Dashboard")

    if not DASHBOARD_OUTPUT_FILE.exists():
        st.warning(
            "AI dashboard output was not found. "
            "Run: python -m src.predict_and_export"
        )
        st.stop()

    ai_df = pd.read_csv(DASHBOARD_OUTPUT_FILE)

    st.metric("Total AI-Scored Reports", len(ai_df))

    st.subheader("Available AI Dashboard Columns")
    st.caption(", ".join(ai_df.columns.tolist()))

    if "review_priority" in ai_df.columns:
        critical_df = ai_df[
            ai_df["review_priority"].astype(str).isin(
                ["Immediate Review", "High Priority", "Manager Review"]
            )
        ].copy()

        st.subheader("Priority Reports")

        st.dataframe(
            critical_df,
            use_container_width=True,
            hide_index=True,
        )

    elif "priority" in ai_df.columns:
        critical_df = ai_df[
            ai_df["priority"].astype(str).isin(["Critical", "High"])
        ].copy()

        st.subheader("Critical / High-Risk Reports")

        st.dataframe(
            critical_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.subheader("All AI-Scored Reports")

        st.dataframe(
            ai_df,
            use_container_width=True,
            hide_index=True,
        )