import onnxruntime as ort
import numpy as np 
import librosa 
from huggingface_hub import hf_hub_download


from fastapi import FastAPI, UploadFile, File

import whisper

import os

from dotenv import load_dotenv

from google import genai

app = FastAPI()

model = whisper.load_model("base")
voice_model_path = hf_hub_download(repo_id="ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model",filename="best_model.onnx")
voice_session = ort.InferenceSession(voice_model_path)

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
def check_voice_authenticity(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    max_len = 48000
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    y = (y - np.mean(y)) / np.sqrt(np.var(y) + 1e-5)
    y = y.astype(np.float32).reshape(1, max_len)
    input_name = voice_session.get_inputs()[0].name
    logits = voice_session.run(None, {input_name: y})[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    return float(probs[0][1])
@app.post("/check-voice")
async def check_voice(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp_voice_check.mp3", "wb") as f:
        f.write(contents)
    fake_probability = check_voice_authenticity("temp_voice_check.mp3")
    return {
        "filename": file.filename,
        "deepfake_probability": round(fake_probability * 100, 2),
        "verdict": "Likely AI/Synthetic Voice" if fake_probability > 0.5 else "Likely Real Human Voice"
    }