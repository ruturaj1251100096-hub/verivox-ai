import onnxruntime as ort
import numpy as np
import librosa

from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model", filename="best_model.onnx")
session = ort.InferenceSession(model_path)

def check_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    max_len = 48000
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    y = (y - np.mean(y)) / np.sqrt(np.var(y) + 1e-5)
    y = y.astype(np.float32).reshape(1, max_len)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: y})[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    fake_probability = probs[0][1]
    print(f"{audio_path}: Deepfake Probability = {fake_probability * 100:.2f}%")

test_files = ["hindi.mp3", "hing.mp3", "mara.mp3", "norm.mp3", "ai_voice.mp3"]
for file in test_files:
    check_audio(file)