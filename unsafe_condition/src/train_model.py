import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

from config import (
    RAW_DATA_FILE, 
    MODELS_DIR, 
    MODEL_FILE, 
    EMBEDDINGS_FILE, 
    EMBEDDING_MODEL_NAME,
    DASHBOARD_OUTPUT_FILE
)

def train_pipeline():
    print("🚀 Starting Unsafe Conditions ML Pipeline...")
    
    # Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("📊 Loading dataset...")
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Filter ONLY for unsafe conditions just to be safe
    df = df[df['report_type'] == 'unsafe conditions'].copy()
    
    # Handle missing text
    df['description_clean'] = df['description_clean'].fillna("No description provided")
    
    # 2. Generate Sentence-BERT Embeddings
    print(f"🧠 Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("⚙️ Converting descriptions to text embeddings (this takes a moment)...")
    X = embedder.encode(df['description_clean'].tolist(), show_progress_bar=True)
    y = df['binary_sif'].values
    
    # Save the embeddings array for Tab 3 (AI Pattern Search) later
    np.save(EMBEDDINGS_FILE, X)
    print("✅ Embeddings saved successfully!")
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train XGBoost Model
    print("🤖 Training XGBoost classifier...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Model Evaluation on Test Data:")
    predictions = xgb_model.predict(X_test)
    print(classification_report(y_test, predictions))
    
    # 5. Save the trained model
    joblib.dump(xgb_model, MODEL_FILE)
    print(f"✅ XGBoost model saved to {MODEL_FILE}")
    
    # 6. Save a clean historical dashboard file for Tab 2
    # Standardizing columns so your previous UI code works flawlessly
    dashboard_df = df[['report_id', 'site', 'activity', 'description_clean', 'binary_sif', 'sif_probability']].copy()
    
    # Map old columns to the new expected names
    dashboard_df = dashboard_df.rename(columns={'description_clean': 'description_raw'})
    
    # Assign priorities based on the existing probability data for the charts
    conditions = [
        dashboard_df['sif_probability'] >= 0.80,
        dashboard_df['sif_probability'] >= 0.60,
        dashboard_df['sif_probability'] >= 0.30
    ]
    choices = ["Critical", "High", "Medium"]
    dashboard_df['priority'] = np.select(conditions, choices, default="Low")
    
    dashboard_df.to_csv(DASHBOARD_OUTPUT_FILE, index=False)
    print(f"✅ Dashboard historical file saved to {DASHBOARD_OUTPUT_FILE}")
    print("🎉 Pipeline complete! Ready for dashboard deployment.")

if __name__ == "__main__":
    train_pipeline()