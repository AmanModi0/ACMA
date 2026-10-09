from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

response = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input="In one sentence, explain what content moderation is."
)

print(response.output_text)