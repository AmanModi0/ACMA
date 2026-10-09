from flask import Flask, request, jsonify, render_template
import re
import os
import json
import base64
import joblib
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer
from google import genai

load_dotenv()

app = Flask(__name__, template_folder="template")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

model = joblib.load("toxicity_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("Warning: GEMINI_API_KEY not found in .env")

gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

sia = SentimentIntensityAnalyzer()



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict.html")
def predict():
    return render_template("predict.html")

@app.route("/aboutus.html")
def aboutus():
    return render_template("aboutus.html")

def clean_json_response(text):
    text = text.strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text

def classify_with_gemini(text, ml_result, ml_confidence, sentiment):
    if not gemini_client:
        return {
            "available": False,
            "decision": ml_result,
            "category": "ML-only",
            "confidence": ml_confidence,
            "explanation": "Gemini API key is not configured."
        }

    prompt = f"""
You are an AI content moderation system.

Analyze the following user-generated text for toxicity, harassment, abuse, threats, hate, insults, profanity, or other harmful language.

Text:
{text}

Traditional ML prediction:
{ml_result}

Traditional ML confidence:
{ml_confidence}

Sentiment:
{sentiment}

Use the Traditional ML prediction as an additional signal, but perform your own contextual analysis of the text.

Return ONLY valid JSON in exactly this format:

{{
    "decision": "Toxic" or "Non-Toxic",
    "category": "Harassment" or "Insult" or "Threat" or "Hate" or "Profanity" or "Other" or "None",
    "confidence": number between 0 and 1,
    "explanation": "short explanation"
}}

Do not add markdown or any text outside the JSON.
"""

    try:
        response = gemini_client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt
        )

        result_text = clean_json_response(response.output_text)
        result = json.loads(result_text)

        return {
            "available": True,
            "decision": result.get("decision", ml_result),
            "category": result.get("category", "Other"),
            "confidence": float(result.get("confidence", ml_confidence)),
            "explanation": result.get(
                "explanation",
                "No explanation provided."
            )
        }

    except Exception as e:
        print("Gemini text error:", str(e))

        return {
            "available": False,
            "decision": ml_result,
            "category": "ML fallback",
            "confidence": ml_confidence,
            "explanation": "Gemini unavailable — ML result used."
        }


def detect_toxicity_in_text(text):
    text_processed = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    ).strip()

    features = vectorizer.transform([text_processed])
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[0][1]

        ml_confidence = float(
            probability if prediction == 1 else 1 - probability
        )
    else:
        ml_confidence = 1.0

    ml_result = "Toxic" if prediction == 1 else "Non-Toxic"

    sentiment_score = sia.polarity_scores(text)["compound"]

    if sentiment_score >= 0.05:
        sentiment_result = "positive"
    elif sentiment_score <= -0.05:
        sentiment_result = "negative"
    else:
        sentiment_result = "neutral"

    gemini_result = classify_with_gemini(
        text,
        ml_result,
        round(ml_confidence, 4),
        sentiment_result
    )

    return {
        "text": text_processed,
        "ml_result": ml_result,
        "ml_confidence": round(ml_confidence, 4),
        "sentiment": sentiment_result,
        "gemini": gemini_result
    }

@app.route("/detect_toxicity", methods=["POST"])
def detect_toxicity():
    text = request.form.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "Please enter text to analyze."
        }), 400

    result = detect_toxicity_in_text(text)

    return jsonify({
        "type": "text",
        "text": result["text"],
        "toxicity_result": result["gemini"]["decision"],
        "ml_prediction": result["ml_result"],
        "ml_confidence": result["ml_confidence"],
        "sentiment": result["sentiment"],
        "gemini_available": result["gemini"]["available"],
        "category": result["gemini"]["category"],
        "gemini_confidence": result["gemini"]["confidence"],
        "explanation": result["gemini"]["explanation"]
    })



if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )