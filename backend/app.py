from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import os
import hashlib
import tempfile
from typing import Optional
import io

try:
    import torch
    from PIL import Image
    from torchvision import transforms
except Exception:
    torch = None
    Image = None
    transforms = None

try:
    import cv2
except Exception:
    cv2 = None

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


MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "models/deepfake_model.pt")
FAKE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", "0.5"))
VIDEO_SAMPLE_FRAMES = max(4, int(os.getenv("VIDEO_SAMPLE_FRAMES", "12")))

model: Optional["torch.nn.Module"] = None
model_device: str = "cpu"
model_error: Optional[str] = None


def _try_load_model() -> None:
    global model, model_device, model_error
    model_error = None

    if torch is None or Image is None or transforms is None:
        model_error = "PyTorch/Pillow/torchvision not available."
        return

    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        model_error = f"Model checkpoint not found at {MODEL_CHECKPOINT_PATH}."
        return

    try:
        model_device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = torch.load(MODEL_CHECKPOINT_PATH, map_location=model_device)
        model_obj = loaded.get("model") if isinstance(loaded, dict) and "model" in loaded else loaded
        if not hasattr(model_obj, "eval"):
            raise ValueError("Checkpoint did not contain a valid torch model object.")

        model_obj.eval()
        model = model_obj.to(model_device)
    except Exception as exc:
        model = None
        model_error = f"Failed to load model: {exc}"


def _has_model_inference() -> bool:
    return model is not None and torch is not None and Image is not None and transforms is not None


def _image_transform() -> "transforms.Compose":
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _output_to_fake_probability(output: "torch.Tensor") -> float:
    out = output.detach().float().flatten()
    if out.numel() == 1:
        prob = torch.sigmoid(out[0]).item()
        return float(prob)
    if out.numel() >= 2:
        probs = torch.softmax(out[:2], dim=0)
        return float(probs[1].item())
    raise ValueError("Unexpected model output shape.")


def _predict_image_with_model(contents: bytes) -> float:
    if not _has_model_inference():
        raise RuntimeError("Model inference is not available.")

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    tensor = _image_transform()(image).unsqueeze(0).to(model_device)
    with torch.no_grad():
        output = model(tensor)
    return _bounded_score(_output_to_fake_probability(output))


def _sample_video_frames(video_path: str, sample_count: int) -> list:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video inference but is not installed.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video for frame sampling.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video appears empty or unreadable.")

    positions = sorted({int(i * (total_frames - 1) / max(1, sample_count - 1)) for i in range(sample_count)})
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        frames.append(pil_frame)

    cap.release()
    if not frames:
        raise ValueError("Failed to sample frames from video.")
    return frames


def _predict_video_with_model(contents: bytes) -> float:
    if not _has_model_inference():
        raise RuntimeError("Model inference is not available.")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        frames = _sample_video_frames(tmp.name, VIDEO_SAMPLE_FRAMES)

    probs = []
    transform = _image_transform()
    with torch.no_grad():
        for frame in frames:
            tensor = transform(frame).unsqueeze(0).to(model_device)
            output = model(tensor)
            probs.append(_output_to_fake_probability(output))

    return _bounded_score(sum(probs) / len(probs))


def _bounded_score(raw_score: float) -> float:
    return max(0.05, min(0.95, raw_score))


def _score_from_bytes(contents: bytes, media_bias: float = 0.0) -> float:
    """
    Deterministic heuristic score from file bytes.
    Not a real deepfake model, but avoids random outputs and gives stable behavior.
    """
    digest = hashlib.sha256(contents).digest()
    digest_component = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    unique_ratio = len(set(contents[:50000])) / 256.0
    size_component = min(len(contents) / 10_000_000, 1.0)

    raw_score = (
        0.48 * digest_component
        + 0.32 * unique_ratio
        + 0.20 * size_component
        + media_bias
    )
    return _bounded_score(raw_score)


_try_load_model()

@app.get("/")
def read_root():
    return {
        "message": "DeepFake Detection API is running.",
        "endpoints": ["/predict/image", "/predict/video", "/health"],
        "inference_mode": "model" if _has_model_inference() else "deterministic-baseline",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "deepfake-api",
        "inference_mode": "model" if _has_model_inference() else "deterministic-baseline",
        "model_checkpoint_path": MODEL_CHECKPOINT_PATH,
        "model_loaded": _has_model_inference(),
        "model_error": model_error,
    }

@app.post("/predict/image", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image bytes to validate upload is not empty.
        contents = await file.read()
        if not contents:
            raise ValueError("Empty image file provided")

        if _has_model_inference():
            fake_score = _predict_image_with_model(contents)
            mode_message = "Prediction complete using model inference."
        else:
            fake_score = _score_from_bytes(contents, media_bias=0.0)
            mode_message = "Prediction complete (deterministic baseline scoring)."

        is_fake = bool(fake_score >= FAKE_THRESHOLD)

        return PredictionResult(
            is_fake=is_fake,
            confidence=round(fake_score if is_fake else 1 - fake_score, 4),
            message=mode_message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/video", response_model=PredictionResult)
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
         raise HTTPException(status_code=400, detail="File provided is not a video.")
    
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Empty video file provided")

        if _has_model_inference() and cv2 is not None:
            fake_score = _predict_video_with_model(contents)
            mode_message = "Video prediction complete using model inference."
        else:
            fake_score = _score_from_bytes(contents, media_bias=0.03)
            mode_message = "Video prediction complete (deterministic baseline scoring)."

        is_fake = bool(fake_score >= FAKE_THRESHOLD)

        return PredictionResult(
            is_fake=is_fake,
            confidence=round(fake_score if is_fake else 1 - fake_score, 4),
            message=mode_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
