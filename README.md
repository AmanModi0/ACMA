<p align="center">
  <img src="./static/img/cc_white.png" alt="AI Content Moderation" width=400/>
</p>
<h1 align=center><b>AI Content Moderation Analysis <i>(ACMA)</i><b></h1>

## Overview
ACMA (AI Content Moderation Analysis) is an advanced AI-driven content moderation system designed to detect and analyze toxicity, harassment, abuse, threats, hate, insults, profanity, and other harmful language in user-generated text. This system helps maintain safe online environments by enforcing community guidelines, legal compliance, and ethical standards while respecting user privacy and freedom of expression.

This project focuses specifically on text moderation, leveraging traditional machine learning alongside modern Large Language Models (LLMs) to provide deep, accurate context analysis.

## Features
### Text-Based Content Analysis
- **Traditional Machine Learning**: Rapidly predicts text toxicity using a Logistic Regression classifier and TF-IDF vectorization.
- **Sentiment Analysis**: Evaluates emotional tone (positive, negative, neutral) using NLTK's VADER lexicon.
- **Advanced LLM Context Analysis**: Leverages the Google Gemini API for deep contextual understanding, categorized tagging (e.g., Hate, Threat, Insult), and detailed explanations of moderation decisions.
- **Real-time Interface**: Clean, modern web UI built with Tailwind CSS for instant content analysis.

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, joblib
- **Natural Language Processing**: NLTK (VADER)
- **Large Language Models**: Google GenAI SDK (Gemini API)
- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **Environment Management**: python-dotenv
- **Deployment**: Gunicorn, optimized for platforms like Render

## Project Workflow
1. **User Input**: User submits text via the web interface.
2. **Preprocessing & ML Prediction**: The text is cleaned and passed through a TF-IDF vectorizer and Logistic Regression model to get a baseline toxicity score and confidence level.
3. **Sentiment Check**: NLTK VADER analyzes the sentiment of the text.
4. **LLM Evaluation**: The original text, ML prediction, and sentiment score are sent to the Gemini API.
5. **Final Decision**: Gemini returns a structured JSON response containing the final decision, specific category, confidence, and explanation.
6. **Display**: Results are rendered dynamically on the frontend.

## Project Structure

```
├── app.py                          # Main Flask application
├── train_toxicity_model.py         # Script to generate the ML models
├── toxicity_classifier.pkl         # Trained toxicity detection model
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer for text processing
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API Keys)
├── template/                       # HTML templates
│   ├── index.html                  # Home page
│   ├── predict.html                # Content analysis interface
│   └── aboutus.html                # About page
└── static/                         # Static assets
    ├── img/                        # Images and icons
    └── styles/                     # CSS stylesheets
```

## Installation and Setup

### Prerequisites
- Python 3.7+
- pip package manager
- Gemini API Key

### Local Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/AmanModi0/ACMA.git
cd ACMA
```

2. **Set up a Virtual Environment (Recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Variables**
Ensure you have a `.env` file in the root directory containing your Gemini API key:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

5. **Run the Flask server**
```bash
python app.py
```
Open your web browser and navigate to `http://localhost:5001`.

## Deployment (Render)

This project is optimized for deployment on Render as a Web Service.

1. Create a new **Web Service** on Render.
2. Link your GitHub repository.
3. Configuration:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
4. Add your `GEMINI_API_KEY` as an Environment Variable in the Render dashboard.

## API Endpoint

The system provides a REST API endpoint for integration:

**POST** `/detect_toxicity`

**Form Data:**
- `text`: Text content to analyze

**Response Format:**
```json
{
  "type": "text",
  "text": "input text",
  "toxicity_result": "Toxic",
  "category": "Insult",
  "gemini_confidence": 0.95,
  "ml_prediction": "Toxic",
  "ml_confidence": 0.88,
  "sentiment": "negative",
  "explanation": "The text contains offensive language.",
  "gemini_available": true
}
```

## Contact Information

**Developer**: Aman Modi <br/>

## License

© 2026-Present ACMA - All rights reserved.
