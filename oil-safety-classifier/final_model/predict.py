import joblib
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_risk(report_text):
    # Load models
    model = joblib.load('unsafe_act_model.pkl')
    vectorizer = joblib.load('unsafe_act_vectorizer.pkl')
    le = joblib.load('unsafe_act_label_encoder.pkl')
    
    # Clean and transform
    cleaned = clean_text(report_text)
    X = vectorizer.transform([cleaned])
    
    # Get prediction
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    risk = le.inverse_transform([pred])[0]  # 0=Non-SIF, 1=SIF
    confidence = max(prob) * 100
    
    # Get all probabilities
    classes = le.classes_
    probs = {}
    for i, cls in enumerate(classes):
        probs[str(cls)] = f"{prob[i] * 100:.1f}%"
    
    # Calculate severity score
    if risk == 1:  # SIF
        score = 75 + (confidence * 0.25)
        classification = "🔴 SIF Potential"
        priority = "CRITICAL"
    else:  # Non-SIF
        score = 5 + (confidence * 0.15)
        classification = "✅ Non-SIF Potential"
        priority = "LOW"
    
    score = min(100, score)
    
    return {
        'sif_binary': risk,
        'risk': 'SIF' if risk == 1 else 'Non-SIF',
        'score': f"{score:.0f}/100",
        'classification': classification,
        'priority': priority,
        'confidence': f"{confidence:.1f}%",
        'probabilities': probs
    }

# Test
if __name__ == "__main__":
    print("="*60)
    print("UNSAFE ACT MODEL - PREDICTIONS")
    print("="*60)
    
    test_reports = [
        "Worker bypassed safety lockout on high-pressure line.",
        "Worker completed daily safety inspection checklist."
    ]
    
    for report in test_reports:
        result = predict_risk(report)
        print(f"\nReport: {report[:50]}...")
        print(f"SIF Binary  : {result['sif_binary']}")
        print(f"Risk Level  : {result['risk']}")
        print(f"Severity Score: {result['score']}")
        print(f"Classification: {result['classification']}")
        print(f"Priority    : {result['priority']}")
        print(f"Confidence  : {result['confidence']}")
        print("-"*40)