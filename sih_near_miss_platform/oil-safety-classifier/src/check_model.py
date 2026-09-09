import joblib

model = joblib.load("models/unsafe_act_model.pkl")
vectorizer = joblib.load("models/unsafe_act_vectorizer.pkl")

print("========================================")
print("MODEL CHECK")
print("========================================")

print("Model classes:")
print(model.classes_)

print("\nNumber of classes:")
print(len(model.classes_))

print("\nNumber of TF-IDF features:")
print(len(vectorizer.get_feature_names_out()))

print("\nMODEL CHECK COMPLETE")
