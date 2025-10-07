# api.py

import os
import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException # Added Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from tensorflow.keras.models import load_model
from detector import TulsiDiseaseDetector

APP_STATIC_DIR = os.path.join(os.path.dirname(__file__), "frontend")

# --- CHANGED: Dictionary to hold all models ---
# NOTE: I've renamed 'tulsi_disease_detection_best_model.h5' to the more descriptive name
# from your file list. Make sure your file is named 'best_VGG16_Transfer_model.h5'.
MODEL_FILES = {
    "vgg16": "best_VGG16_Transfer_model.h5",
    "mobilenet": "best_MobileNet_Transfer_model.h5",
    "custom_cnn": "best_Custom_CNN_model.h5",
}

# --- CHANGED: This will hold multiple detector instances ---
detectors = {}
config_path = ""

app = FastAPI(title="Tulsi Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_model_files_exist():
    global config_path
    config_path = os.path.join(os.path.dirname(__file__), "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("model_config.json not found in project root")

    # --- CHANGED: Check for all model files ---
    for model_key, model_file in MODEL_FILES.items():
        model_path = os.path.join(os.path.dirname(__file__), model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_file} not found for model '{model_key}'")


@app.on_event("startup")
def load_detectors_on_startup():
    # --- CHANGED: Load all models into the 'detectors' dictionary ---
    global detectors
    _ensure_model_files_exist()
    print("Loading models...")
    for model_key, model_file in MODEL_FILES.items():
        model_path = os.path.join(os.path.dirname(__file__), model_file)
        print(f" -> Loading '{model_key}' from {model_file}")
        detectors[model_key] = TulsiDiseaseDetector(model_path, config_path)
    print("All models loaded successfully!")


# --- NEW: Endpoint to list available models ---
@app.get("/models")
def get_models():
    # Returns a list of model keys for the frontend dropdown
    return {"models": list(detectors.keys())}


@app.get("/health")
def health_check():
    return {"status": "ok"}


# --- CHANGED: The predict function now accepts a model name ---
@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    if model_name not in detectors:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        temp_dir = os.path.join(os.path.dirname(__file__), "_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # --- CHANGED: Use the selected detector ---
        selected_detector = detectors[model_name]
        result = selected_detector.predict(temp_path)
        recommendation = selected_detector.get_treatment_recommendation(result)

        response = {
            "prediction": result,
            "recommendation": recommendation,
            "model_used": model_name, # Also return which model was used
        }
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if os.path.isdir(APP_STATIC_DIR):
    app.mount("/", StaticFiles(directory=APP_STATIC_DIR, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)