import io
import os
import time
import yaml
import torch
import numpy as np
import librosa
from PIL import Image
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms
from prometheus_client import Counter, Histogram, start_http_server

# Import model architecture definition
from export_model import create_resnext_model

app = FastAPI(title="Dementia Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
PREDICTION_COUNT = Counter('prediction_count', 'Number of predictions made', ['result'])
PREDICTION_TIME = Histogram('prediction_time', 'Time taken for prediction')

# Load model info
with open("exported_model/model_info.yaml", "r") as f:
    model_info = yaml.safe_load(f)

# Response model
class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    inference_time: float
    model_info: Dict[str, Any]


class AudioConfig(BaseModel):
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 8000

# Load model (global variables)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = model_info["class_names"]

# Preprocessing transforms - same as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.on_event("startup")
def load_model():
    global model
    model_path = "exported_model/model.pt"  # Use TorchScript model
    
    if os.path.exists(model_path):
        try:
            # Load TorchScript model
            model = torch.jit.load(model_path, map_location=device)
            print(f"Model loaded successfully on {device}")
            
            # Start Prometheus metrics server
            start_http_server(8000)
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to regular PyTorch model if TorchScript fails
            model = create_resnext_model()
            model.load_state_dict(torch.load("best_resnext.pth", map_location=device))
            model.eval()
    else:
        # Fallback to the original .pth file
        model = create_resnext_model()
        model.load_state_dict(torch.load("best_resnext.pth", map_location=device))
        model.eval()

@app.get("/")
def health_check():
    return {"status": "healthy", "model": model_info["name"]}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    
    # Start prediction process
    start_time = time.time()
    
    # Read image
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        with PREDICTION_TIME.time():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class_idx = probabilities.argmax().item()
    
    # Create response
    class_name = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()
    inference_time = time.time() - start_time
    
    # Update Prometheus metrics
    PREDICTION_COUNT.labels(result=class_name).inc()
    
    return PredictionResponse(
        filename=file.filename,
        prediction=class_name,
        confidence=float(confidence),
        inference_time=inference_time,
        model_info=model_info
    )

@app.get("/model-info")
def get_model_info():
    return model_info

@app.post("/predict-audio", response_model=PredictionResponse)
async def predict_audio(
    file: UploadFile = File(...),
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: int = 20,
    fmax: int = 8000
):
    """
    Process audio files by converting them to mel spectrograms and running prediction on them.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an audio file")
    
    # Create config from parameters
    config = {
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "fmin": fmin,
        "fmax": fmax
    }
    
    # Start prediction process
    start_time = time.time()
    
    # Read audio file
    content = await file.read()
    audio_bytes = io.BytesIO(content)
    
    try:
        # Load audio using librosa
        y, sr = librosa.load(audio_bytes, sr=None)
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"]
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 255] for image conversion
        mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
        
        # Convert to RGB image (model expects 3 channels)
        img = Image.fromarray(mel_spec_norm).convert('RGB')
        
        # Apply the same transform used for images
        image_tensor = transform(img).unsqueeze(0).to(device)
        
        # Inference
        with PREDICTION_TIME.time():
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class_idx = probabilities.argmax().item()
        
        # Create response
        class_name = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()
        inference_time = time.time() - start_time
        
        # Update Prometheus metrics
        PREDICTION_COUNT.labels(result=class_name).inc()
        
        return PredictionResponse(
            filename=file.filename,
            prediction=class_name,
            confidence=float(confidence),
            inference_time=inference_time,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)