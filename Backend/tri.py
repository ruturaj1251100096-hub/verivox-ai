from fastapi import FastAPI, UploadFile, File

import whisper

import os

from dotenv import load_dotenv

from google import genai

app = FastAPI()

model = whisper.load_model("base")

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")

def read_root():

    return {"message": "Verivox-ai backend is running!"}

@app.post("/analyze-call")

async def analyze_call(file: UploadFile = File(...)):

    contents = await file.read()

    with open("temp_audio.mp3", "wb") as f:

        f.write(contents)

    result = model.transcribe("temp_audio.mp3")

    return {

        "filename": file.filename,

        "transcript": result["text"],

        "detected_language": result["language"]

    }
@app.post("/detect-scam")
async def detect_scam(transcript: str):
    prompt = f"Analyze this call transcript for scam signals (OTP requests, urgency, threats, bank impersonation, fake authority). Transcript: {transcript}. Respond with: risk_level (Safe/Suspicious/Critical), reasons (list), and a one-line explanation."
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )
        return {"analysis": response.text}
    except Exception as e:
        print(f"Gemini API error: {e}")
        return {
            "analysis": "risk_level: Unable to complete analysis right now (temporary service issue). Please try again in a moment.",
            "error": True
        }