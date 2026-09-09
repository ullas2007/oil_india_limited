from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import speech_recognition as sr  # NEW: For Voice-to-Text

from src.config import (
    AUDIO_UPLOAD_DIR,
    BASE_DIR,
    IMAGE_UPLOAD_DIR,
    SUBMISSIONS_CSV,
    create_project_directories,
)


create_project_directories()


st.set_page_config(
    page_title="Employee Near-Miss Reporting",
    page_icon="🦺",
    layout="wide",
)


def safe_file_name(file_name: str) -> str:
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "._-"
    )

    return "".join(
        character if character in allowed else "_"
        for character in file_name
    )


def save_uploaded_file(uploaded_file, folder: Path, report_id: str, file_type: str):
    if uploaded_file is None:
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = safe_file_name(uploaded_file.name)

    saved_name = f"{report_id}_{file_type}_{timestamp}_{original_name}"
    saved_path = folder / saved_name

    # Reset file pointer just in case it was read by the speech recognizer
    uploaded_file.seek(0)
    saved_path.write_bytes(uploaded_file.getbuffer())

    # Save relative path in CSV, not a full Windows path
    return str(saved_path.relative_to(BASE_DIR))


def load_submissions():
    if SUBMISSIONS_CSV.exists():
        return pd.read_csv(SUBMISSIONS_CSV)

    return pd.DataFrame()


def append_submission(submission: dict):
    new_row = pd.DataFrame([submission])

    if SUBMISSIONS_CSV.exists():
        old_df = pd.read_csv(SUBMISSIONS_CSV)
        updated_df = pd.concat(
            [old_df, new_row],
            ignore_index=True,
        )
    else:
        updated_df = new_row

    updated_df.to_csv(SUBMISSIONS_CSV, index=False)


# --- INITIALIZE SESSION STATE FOR VOICE TEXT ---
if "transcribed_text" not in st.session_state:
    st.session_state["transcribed_text"] = ""
if "last_processed_audio" not in st.session_state:
    st.session_state["last_processed_audio"] = None


st.title("🦺 Employee Near-Miss Reporting")
st.caption(
    "Report a near miss, unsafe condition, or unsafe act for HSE review."
)

st.info(
    "Submit the report as soon as possible. "
    "A near miss is an event that did not cause injury or damage, "
    "but could have caused harm under slightly different circumstances."
)

# ==========================================
# STEP 1: VOICE RECORDER (Placed outside the form to allow live updates)
# ==========================================
st.subheader("🎤 Step 1: Voice-to-Text (Optional)")
st.write("Record your hazard description here, and our system will type it out for you!")

audio_file = st.audio_input("Record voice description")

# If a new audio file is recorded, transcribe it instantly!
if audio_file is not None and audio_file != st.session_state["last_processed_audio"]:
    with st.spinner("Converting your voice to text..."):
        try:
            recognizer = sr.Recognizer()
            # Streamlit audio input gives us a WAV file buffer, perfect for SpeechRecognition
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                
                # Save the text and update state so we don't process the same audio twice
                st.session_state["transcribed_text"] = text
                st.session_state["last_processed_audio"] = audio_file
                st.success("Audio transcribed successfully! You can review and edit it below.")
        except sr.UnknownValueError:
            st.warning("Could not clearly understand the audio. Please type the description manually.")
        except sr.RequestError:
            st.error("Internet connection issue. Could not reach speech-to-text service.")
        except Exception as e:
            st.error("Something went wrong with the audio. Please type manually.")


# ==========================================
# STEP 2: FORM SUBMISSION
# ==========================================
st.subheader("📝 Step 2: Fill Report Details")

with st.form("employee_near_miss_form", clear_on_submit=False):
    
    col1, col2 = st.columns(2)

    with col1:
        employee_name = st.text_input(
            "Employee name",
            placeholder="Example: Priya Sharma",
        )

        report_type = st.selectbox(
            "Report type",
            [
                "Near Miss",
                "Unsafe Condition",
                "Unsafe Act",
            ],
        )

        site = st.selectbox(
            "Site / location",
            ["Tank Farm A", "Tank Farm B", "Main Plant", "Warehouse", "Loading Dock"]
        )

        unit = st.selectbox(
            "Unit / department",
            ["Tank Farm", "Processing", "Logistics", "Maintenance", "Electrical"]
        )

    with col2:
        activity = st.selectbox(
            "Activity being performed",
            ["Welding", "Inspection", "Cleaning", "Heavy Lifting", "Electrical Work", "Other"]
        )

        equipment = st.selectbox(
            "Equipment / area",
            ["Oily-water drain", "Pump Station 1", "Main Boiler", "High Voltage Panel", "Forklift", "Other"]
        )

        reporter_type = st.selectbox(
            "Reporter type",
            [
                "Employee",
                "Contractor",
                "Supervisor",
                "Anonymous",
            ],
        )

        language = st.selectbox(
            "Report language",
            [
                "English",
                "Hindi",
                "Kannada",
                "Tamil",
                "Telugu",
            ],
        )

    # Automatically loads the transcribed text from the microphone!
    description = st.text_area(
        "Describe the near-miss event",
        value=st.session_state["transcribed_text"],  # Pre-fills with Voice Text!
        placeholder=(
            "Explain what happened, where it happened, what unsafe condition "
            "was observed, and what could have happened."
        ),
        height=170,
    )

    immediate_action = st.text_area(
        "Immediate action taken",
        placeholder=(
            "Example: Stopped work, barricaded the area, "
            "informed supervisor, or removed the obstruction."
        ),
        height=100,
    )

    st.subheader("Optional evidence")

    image_file = st.file_uploader(
        "Upload photo",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo of the hazard or unsafe condition.",
    )

    confirmation = st.checkbox(
        "I confirm that this report contains genuine safety information."
    )

    submit_button = st.form_submit_button(
        "Submit Near-Miss Report",
        type="primary",
    )


if submit_button:
    if not confirmation:
        st.error("Please confirm the declaration before submitting the report.")
        st.stop()

    if not site.strip():
        st.error("Site / location is required.")
        st.stop()

    if not description.strip():
        st.error("Please describe the near-miss event.")
        st.stop()

    report_id = "EMP-" + datetime.now().strftime("%Y%m%d%H%M%S")

    image_path = save_uploaded_file(
        image_file,
        IMAGE_UPLOAD_DIR,
        report_id,
        "image",
    )

    # Saves the audio file recorded in Step 1
    audio_path = save_uploaded_file(
        audio_file,
        AUDIO_UPLOAD_DIR,
        report_id,
        "audio",
    )

    submission = {
        "report_id": report_id,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "employee_name": employee_name.strip() if employee_name.strip() else "Anonymous",
        "employee_id": "N/A",  
        "report_type": report_type,
        "site": site.strip(),
        "unit": unit.strip(),
        "activity": activity.strip(),
        "equipment": equipment.strip(),
        "reporter_type": reporter_type,
        "language": language,
        "description": description.strip(),
        "immediate_action": immediate_action.strip(),
        "image_path": image_path,
        "audio_path": audio_path,
        "status": "Submitted",
        "manager_review": "Pending",
        "manager_comment": "",
    }

    append_submission(submission)

    # Clear the transcribed text state so the next report starts fresh
    st.session_state["transcribed_text"] = ""
    st.session_state["last_processed_audio"] = None

    st.success("Near-miss report submitted successfully.")
    st.info(f"Your Report ID: {report_id}")

    st.session_state["last_report_id"] = report_id


st.divider()
st.subheader("Submitted Reports")

submissions_df = load_submissions()

if submissions_df.empty:
    st.info("No employee reports have been submitted yet.")
else:
    total_reports = len(submissions_df)
    pending_reports = len(
        submissions_df[
            submissions_df["manager_review"].astype(str) == "Pending"
        ]
    )

    metric1, metric2 = st.columns(2)
    metric1.metric("Total Submitted Reports", total_reports)
    metric2.metric("Pending Manager Review", pending_reports)

    display_columns = [
        "report_id",
        "submitted_at",
        "report_type",
        "site",
        "activity",
        "status",
        "manager_review",
    ]

    available_columns = [
        column for column in display_columns if column in submissions_df.columns
    ]

    st.dataframe(
        submissions_df[available_columns].sort_values(
            "submitted_at",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )