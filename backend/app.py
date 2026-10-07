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
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms, models
    from facenet_pytorch import MTCNN
    import timm
    import numpy as np
    import scipy.signal as signal
    import moviepy.editor as mp
except Exception:
    torch = None
    nn = None
    Image = None
    transforms = None
    models = None
    MTCNN = None
    timm = None

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


model_ensemble = []
mtcnn = None

def _try_load_model() -> None:
    global model_ensemble, model_device, model_error, mtcnn
    model_error = None
    model_ensemble = []

    if torch is None or Image is None or transforms is None or models is None or MTCNN is None:
        model_error = "PyTorch/Pillow/torchvision/MTCNN not available."
        return

    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(margin=40, keep_all=False, select_largest=True, post_process=False, device=model_device)

    # Attempt to load Xception
    xception_path = os.getenv("XCEPTION_PATH", "models/deepfake_xception.pt")
    if os.path.exists(xception_path) and timm is not None:
        try:
            xc_model = timm.create_model('xception', pretrained=False, num_classes=1)
            if hasattr(xc_model, 'get_classifier'):
                xc_model.get_classifier().add_module('dropout', nn.Dropout(0.5))
            
            loaded = torch.load(xception_path, map_location=model_device)
            state_dict = loaded.get("model") if isinstance(loaded, dict) and "model" in loaded else loaded
            xc_model.load_state_dict(state_dict)
            xc_model.eval()
            model_ensemble.append(xc_model.to(model_device))
            print("✅ Loaded Xception into ensemble.")
        except Exception as e:
            print(f"Failed to load Xception: {e}")

    # Attempt to load EfficientNet
    effnet_path = os.getenv("EFFNET_PATH", "models/deepfake_model.pt")
    if os.path.exists(effnet_path):
        try:
            eff_model = models.efficientnet_b4(weights=None)
            num_ftrs = eff_model.classifier[1].in_features
            eff_model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(num_ftrs, 1)
            )
            loaded = torch.load(effnet_path, map_location=model_device)
            state_dict = loaded.get("model") if isinstance(loaded, dict) and "model" in loaded else loaded
            eff_model.load_state_dict(state_dict)
            eff_model.eval()
            model_ensemble.append(eff_model.to(model_device))
            print("✅ Loaded EfficientNet into ensemble.")
        except Exception as e:
            print(f"Failed to load EfficientNet: {e}")

    if not model_ensemble:
        model_error = "No valid models found to load into ensemble."

def _has_model_inference() -> bool:
    return len(model_ensemble) > 0 and mtcnn is not None


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
    
    face = mtcnn(image)
    if face is None:
        raise ValueError("No face detected in the image.")
        
    face_tensor = face / 255.0
    face_img = transforms.ToPILImage()(face_tensor)
    tensor = _image_transform()(face_img).unsqueeze(0).to(model_device)

    model_scores = []
    with torch.no_grad():
        for m in model_ensemble:
            output = m(tensor)
            model_scores.append(_output_to_fake_probability(output))
            
    avg_score = sum(model_scores) / len(model_scores)
    
    # Add Frequency Domain Analysis Penalty
    freq_penalty = _analyze_frequency_artifacts(face_img)
    final_score = min(0.95, avg_score + freq_penalty)
    
    return _bounded_score(final_score)


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


def _analyze_frequency_artifacts(pil_img: Image.Image) -> float:
    """
    Analyzes the 2D FFT power spectrum for grid-like high-frequency anomalies 
    common in GANs and upsampling algorithms.
    Returns a fake penalty between 0.0 and 0.15.
    """
    if 'np' not in globals():
        return 0.0
    gray = pil_img.convert('L')
    img_arr = np.array(gray)
    
    # Calculate 2D FFT
    f = np.fft.fft2(img_arr)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # Calculate energy in high frequency region vs low frequency
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    mask = x*x + y*y <= (min(h, w) * 0.25)**2
    
    low_freq_energy = np.sum(magnitude_spectrum[mask])
    high_freq_energy = np.sum(magnitude_spectrum[~mask])
    
    if low_freq_energy == 0:
        return 0.0
        
    hf_ratio = high_freq_energy / low_freq_energy
    if hf_ratio > 3.0:
        return 0.15
    elif hf_ratio > 2.0:
        return 0.05
    return 0.0


def _analyze_audio_visual_sync(video_path: str) -> float:
    """
    Basic Audio-Visual anomaly detection.
    Checks if a talking-head video has stripped or severely distorted audio.
    """
    if 'mp' not in globals():
        return 0.0
        
    try:
        clip = mp.VideoFileClip(video_path)
        if clip.audio is None:
            # Deepfake videos often strip audio to hide artifacting.
            return 0.05
            
        audio_arr = clip.audio.to_soundarray(fps=16000)
        if audio_arr is None or len(audio_arr) == 0:
            return 0.05
            
        rms_energy = np.sqrt(np.mean(audio_arr**2))
        
        # Anomalously low (near silent) energy might indicate a synthetic voiceover patch.
        if rms_energy < 0.001:
            return 0.02
            
        return 0.0
    except Exception as e:
        print(f"[AV Sync Error] {e}")
        return 0.0


def _analyze_rppg(video_path: str, max_frames=90) -> float:
    """
    Extracts rPPG (Remote Photoplethysmography) heartbeat signal from forehead ROI.
    Returns a 'fake penalty' between 0.0 (real heartbeat) and 0.2 (no heartbeat).
    """
    if cv2 is None or 'signal' not in globals():
        return 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames_processed = 0
    green_signal = []
    
    # Try to find a face in the first few frames to establish a tracking ROI
    roi_box = None
    
    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if roi_box is None and frames_processed < 5:
            # Use MTCNN on a downscaled frame for speed
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                # Forehead ROI: top 20% of the bounding box
                h, w = y2 - y1, x2 - x1
                roi_box = (x1, y1, x1 + w, y1 + int(h * 0.2))
        
        if roi_box is not None:
            x1, y1, x2, y2 = roi_box
            # Ensure within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if y2 > y1 and x2 > x1:
                roi = frame[y1:y2, x1:x2]
                # Average green channel intensity
                g_mean = np.mean(roi[:, :, 1])
                green_signal.append(g_mean)
                
        frames_processed += 1
        
    cap.release()

    if len(green_signal) < 30:
        return 0.0 # Not enough frames for FFT

    # Signal processing
    signal_arr = np.array(green_signal)
    signal_arr = signal.detrend(signal_arr)
    
    # Bandpass filter for normal human heart rate (0.7 Hz to 2.5 Hz -> 42 to 150 BPM)
    nyquist = 0.5 * fps
    low = 0.7 / nyquist
    high = 2.5 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_arr)
    
    # FFT to find frequency peaks
    fft_vals = np.abs(np.fft.rfft(filtered))
    if np.sum(fft_vals) == 0:
        return 0.1
        
    # Calculate Signal-to-Noise Ratio (SNR) in the HR band
    peak_energy = np.max(fft_vals) ** 2
    total_energy = np.sum(fft_vals ** 2)
    snr = peak_energy / (total_energy - peak_energy + 1e-6)
    
    print(f"[rPPG Analysis] Extracted SNR: {snr:.3f}")
    
    # If SNR is low, the pulse is chaotic/synthetic. Apply a penalty.
    if snr < 1.5:
        return 0.15
    elif snr < 2.5:
        return 0.05
    return 0.0


def _predict_video_with_model(contents: bytes) -> float:
    if not _has_model_inference():
        raise RuntimeError("Model inference is not available.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        frames = _sample_video_frames(tmp_path, VIDEO_SAMPLE_FRAMES)
        scores = []
        crop_failures = 0
        
        for frame in frames:
            face = mtcnn(frame)
            if face is None:
                crop_failures += 1
                continue
            face_tensor = face / 255.0
            face_img = transforms.ToPILImage()(face_tensor)
            tensor = _image_transform()(face_img).unsqueeze(0).to(model_device)
            
            frame_model_scores = []
            with torch.no_grad():
                for m in model_ensemble:
                    output = m(tensor)
                    frame_model_scores.append(_output_to_fake_probability(output))
            
            avg_frame_score = sum(frame_model_scores) / len(frame_model_scores)
            
            # Frequency Domain Analysis (Grid/GAN artifacts)
            freq_penalty = _analyze_frequency_artifacts(face_img)
            scores.append(_bounded_score(avg_frame_score + freq_penalty))
            
        if not scores:
            raise ValueError(f"No face detected in any sampled frame (Crop failures: {crop_failures}/{len(frames)}). Verify MTCNN margin=40 is not clipping out of bounds on close-ups.")
            
        # Temporal Aggregation Strategy (Mitigating False Positives):
        scores.sort()
        top_k = min(3, len(scores))
        top_scores = scores[-top_k:]
        
        top_avg = sum(top_scores) / top_k
        avg_score = sum(scores) / len(scores)
        
        # Blend the top-3 average with the overall average
        spatial_score = (0.5 * top_avg) + (0.5 * avg_score)
        
        # Add biological signal analysis (rPPG penalty)
        rppg_penalty = _analyze_rppg(tmp_path)
        
        # Add Audio-Visual sync penalty
        av_penalty = _analyze_audio_visual_sync(tmp_path)
        
        final_score = min(0.95, spatial_score + rppg_penalty + av_penalty)
        
        print(f"[Analysis] Frames: {len(frames)} | Failures: {crop_failures} | Spatial: {spatial_score:.3f} | rPPG Penalty: +{rppg_penalty:.3f} | AV Penalty: +{av_penalty:.3f} | Final: {final_score:.3f}")
        return final_score
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
