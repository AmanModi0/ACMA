import re
import joblib

vectorizer=joblib.load("tfidf_vectorizer.pkl")
model=joblib.load("toxicity_classifier.pkl")

def predict(text):
    text=re.sub(r"[^a-zA-Z0-9\s]","",text).strip()
    features=vectorizer.transform([text])
    prediction=model.predict(features)[0]
    probability=model.predict_proba(features)[0][1]
    return {
        "label":"Toxic" if prediction==1 else "Non-Toxic",
        "confidence":round(float(probability if prediction==1 else 1-probability),4)
    }

if __name__=="__main__":
    for text in [
        "Thank you for your help with this project",
        "You are a stupid idiot and I hate you"
    ]:
        print(text,"->",predict(text))
