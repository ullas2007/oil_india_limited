import streamlit as st
import pandas as pd
from datetime import datetime
import uuid
import os
from pathlib import Path
import speech_recognition as sr
from src.config import SUBMISSIONS_CSV

def run():
    st.title("👷 Unsafe Condition Reporting Portal")
    st.caption("Report workplace hazards safely and securely.")

    with st.form("employee_report_form"):
        st.subheader("New Unsafe Condition Report")
        
        col1, col2 = st.columns(2)
        with col1:
            site = st.selectbox(
                "Site Location", 
                ["Tank Farm A", "Digboi Refinery", "Moran Oilfield", "Duliajan Oilfield"]
            )
            unit = st.selectbox(
                "Specific Unit / Area", 
                ["Crude Distillation Unit (CDU)", "33kV Substation", "Wellhead Cluster", 
                 "Storage Tank Area", "Offshore Rig Deck", "Admin Building", "Other"]
            )
            
        with col2:
            activity = st.selectbox(
                "Current Activity", 
                ["Maintenance", "Inspection", "Hot Work", "Lifting", "General Walking"]
            )
            immediate_action = st.selectbox(
                "Immediate action taken:", 
                ["Blocked/Barricaded area", "Halted operations", "Notified Supervisor", 
                 "Applied Lockout/Tagout (LOTO)", "Cleaned/Cleared hazard", "None"]
            )
        
        st.write("**Hazard Details**")
        description = st.text_area("Describe the unsafe condition (Optional if using Voice):", height=100)
        
        st.write("**Evidence Upload (Optional)**")
        image_file = st.file_uploader("Upload Photo Evidence (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
        audio_file = st.audio_input("Record Voice Description")
        
        submitted = st.form_submit_button("Submit Report", type="primary")

    if submitted:
        if not description and not audio_file:
            st.error("Please provide either a text description or a voice recording.")
        else:
            with st.spinner("Processing report and extracting voice to text..."):
                report_id = f"UC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
                
                # --- MEDIA SAVING & SPEECH RECOGNITION ---
                media_dir = "data/media"
                os.makedirs(media_dir, exist_ok=True)
                
                saved_image_path = ""
                if image_file:
                    saved_image_path = f"{media_dir}/{report_id}.jpg"
                    with open(saved_image_path, "wb") as f:
                        f.write(image_file.getbuffer())

                saved_audio_path = ""
                audio_transcript = ""
                
                if audio_file:
                    saved_audio_path = f"{media_dir}/{report_id}.wav"
                    with open(saved_audio_path, "wb") as f:
                        f.write(audio_file.getbuffer())
                    
                    # Convert Voice to Text
                    recognizer = sr.Recognizer()
                    try:
                        with sr.AudioFile(saved_audio_path) as source:
                            audio_data = recognizer.record(source)
                            # Using free Google Speech Recognition
                            audio_transcript = recognizer.recognize_google(audio_data)
                    except Exception as e:
                        audio_transcript = f"[Voice recognition failed or offline: {e}]"

                # Combine typed text and spoken text so XGBoost has the full picture
                final_description = description
                if audio_transcript:
                    if final_description:
                        final_description += f"\n\n[Voice Transcript]: {audio_transcript}"
                    else:
                        final_description = audio_transcript

                # Save everything to the CSV
                new_data = pd.DataFrame([{
                    "report_id": report_id,
                    "site": site,
                    "unit": unit,
                    "activity": activity,
                    "description": final_description, # XGBoost will read this!
                    "immediate_action": immediate_action,
                    "image_evidence": saved_image_path,
                    "audio_evidence": saved_audio_path,
                    "status": "Submitted"
                }])
                
                if Path(SUBMISSIONS_CSV).exists():
                    df = pd.read_csv(SUBMISSIONS_CSV)
                    df = pd.concat([df, new_data], ignore_index=True)
                else:
                    df = new_data
                    
                df.to_csv(SUBMISSIONS_CSV, index=False)
                st.success(f"✅ Hazard reported successfully! Tracking ID: {report_id}")