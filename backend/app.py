from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import os
import random

app = FastAPI(title="DeepFake Detection API", description="API to predict if an image or video is a Deepfake.")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

# Configure CORS for the frontend and extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PredictionResult(BaseModel):
    is_fake: bool
    confidence: float
    message: str

@app.get("/")
def read_root():
    return {
        "message": "DeepFake Detection API is running.",
        "endpoints": ["/predict/image", "/predict/video", "/health"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "deepfake-api"}

@app.post("/predict/image", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image bytes to validate upload is not empty.
        contents = await file.read()
        if not contents:
            raise ValueError("Empty image file provided")

        # Mock prediction logic (replace with actual EfficientNet + MTCNN logic)
        # For demonstration, we'll assign a random score
        # In a real scenario, you'd run MTCNN to crop the face, then pass to the model
        fake_score = random.random()
        is_fake = bool(fake_score > 0.5)

        return PredictionResult(
            is_fake=is_fake,
            confidence=round(fake_score if is_fake else 1 - fake_score, 4),
            message="Prediction complete."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/video", response_model=PredictionResult)
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
         raise HTTPException(status_code=400, detail="File provided is not a video.")
    
    try:
        # For mocked backend, we will just return a fake result
        fake_score = random.random()
        is_fake = bool(fake_score > 0.5)

        return PredictionResult(
            is_fake=is_fake,
            confidence=round(fake_score if is_fake else 1 - fake_score, 4),
            message="Video prediction complete. (Mocked)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
