from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Verivox-ai backend is running!"}
from fastapi import UploadFile, File

@app.post("/analyze-call")
async def analyze_call(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "size_in_bytes": len(contents),
        "message": "Audio received successfully!"
    }