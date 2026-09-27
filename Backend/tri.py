from fastapi import FastAPI, UploadFile, File
import whisper

app = FastAPI()
model = whisper.load_model("base")
@app.get("/")
def read_root():
    return {"message": "Verivox-ai backend is running!"}
from fastapi import UploadFile, File

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