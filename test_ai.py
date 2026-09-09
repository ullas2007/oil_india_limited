import joblib
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Only import BASE_DIR from your config
from src.config import BASE_DIR

# Hardcode the thresholds and paths so it doesn't crash!
CRITICAL_THRESHOLD = 80
HIGH_THRESHOLD = 60
MEDIUM_THRESHOLD = 30
SIF_MODEL_FILE = BASE_DIR / "models" / "xgboost_near_miss_model.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

print("\n⚙️  Waking up the AI Brain (Loading models)...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

if SIF_MODEL_FILE.exists():
    model = joblib.load(SIF_MODEL_FILE)
    print("✅ AI Ready!\n")
else:
    print(f"❌ Error: Model not found at {SIF_MODEL_FILE}")
    exit()

# Terminal colors for a cool hacker look during your presentation!
RED = '\033[91m'
ORANGE = '\033[93m'
YELLOW = '\033[33m'
GREEN = '\033[92m'
RESET = '\033[0m'

print("="*50)
print("🏭 NEAR-MISS / UNSAFE ACT TERMINAL TESTER")
print("Type 'exit' or 'quit' to close the program.")
print("="*50)

# Interactive Loop
while True:
    description = input("\n📝 Enter hazard description:\n> ")
    
    if description.lower() in ['exit', 'quit']:
        print("Closing tester. Goodbye!")
        break
        
    if not description.strip():
        continue
        
    # AI calculates the risk
    embedding = embedder.encode([description])
    prob_decimal = model.predict_proba(embedding)[0][1]
    score = round(prob_decimal * 100, 1)
    
    # Check thresholds
    if score >= CRITICAL_THRESHOLD:
        priority = "Critical"
        color = RED
    elif score >= HIGH_THRESHOLD:
        priority = "High"
        color = ORANGE
    elif score >= MEDIUM_THRESHOLD:
        priority = "Medium"
        color = YELLOW
    else:
        priority = "Low"
        color = GREEN
        
    # Print Results
    print(f"\n   📊 AI Score:  {score}%")
    print(f"   🚨 Risk Level: {color}{priority}{RESET}")
    print("-" * 50)