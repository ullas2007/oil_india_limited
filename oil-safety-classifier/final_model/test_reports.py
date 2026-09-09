import joblib
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(report_text):
    model = joblib.load('unsafe_act_model.pkl')
    vectorizer = joblib.load('unsafe_act_vectorizer.pkl')
    le = joblib.load('unsafe_act_label_encoder.pkl')
    
    cleaned = clean_text(report_text)
    X = vectorizer.transform([cleaned])
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    risk = le.inverse_transform([pred])[0]
    confidence = max(prob) * 100
    
    if risk == 1:
        score = 75 + (confidence * 0.25)
        classification = "🔴 SIF Potential"
        priority = "CRITICAL"
    else:
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
        'confidence': f"{confidence:.1f}%"
    }

print("="*60)
print("COMPLETE TEST - SIF vs Non-SIF")
print("="*60)

# Test reports with expected sif_binary (1=SIF, 0=Non-SIF)
reports = [
    # SIF (should predict 1)
    ("Worker bypassed safety lockout on high-pressure line.", 1),
    ("Operator worked at height without fall protection.", 1),
    ("Worker entered confined space without gas testing.", 1),
    ("Welding performed near flammable materials without fire watch.", 1),
    
    # Non-SIF (should predict 0)
    ("Worker completed daily safety inspection checklist.", 0),
    ("Worker reported minor housekeeping issue and cleaned it.", 0),
    ("Worker wore safety glasses correctly during operation.", 0),
    ("Routine shift handover completed. No hazards reported.", 0)
]

correct = 0
for i, (report, expected) in enumerate(reports, 1):
    result = predict(report)
    is_correct = result['sif_binary'] == expected
    if is_correct:
        correct += 1
    
    status = "✅" if is_correct else "❌"
    print(f"\n{status} Test {i}:")
    print(f"Expected: {expected} (1=SIF, 0=Non-SIF)")
    print(f"Predicted: {result['sif_binary']} ({result['risk']})")
    print(f"Score: {result['score']}")
    print(f"Classification: {result['classification']}")
    print(f"Priority: {result['priority']}")
    print(f"Confidence: {result['confidence']}")

print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {correct}/{len(reports)} ({correct/len(reports)*100:.1f}%)")
print("="*60)