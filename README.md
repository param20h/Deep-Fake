# DeepFake Detector

A comprehensive deepfake detection system with multiple interfaces: a web application, browser extension, and REST API backend.

## 🎯 Features

- **Image & Video Analysis**: Detect AI-generated or manipulated faces in images and video files
- **Browser Extension**: Instantly scan images on any website with a single click
- **Web Dashboard**: User-friendly interface for uploading and analyzing content
- **REST API**: Programmatic access for integration into other applications
- **Real-time Detection**: Fast processing with configurable confidence thresholds

## 📁 Project Structure

```
deepfake/
├── backend/          # FastAPI server with ML detection models
├── frontend/         # Next.js web application
├── extension/        # Chrome/Chromium browser extension
└── README.md        # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (for backend)
- **Node.js 18+** (for frontend)
- **npm or yarn** (for package management)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# For ML features, also install:
pip install -r requirements-ml.txt
```

3. Set environment variables (optional):
```bash
export ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"
export MODEL_CHECKPOINT_PATH="models/deepfake_model.pt"
export FAKE_THRESHOLD="0.5"
export VIDEO_SAMPLE_FRAMES="12"
```

4. Run the API server:
```bash
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env.local` file:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

4. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Browser Extension Setup

1. Navigate to the extension directory:
```bash
cd extension
```

2. Load the extension in your browser:
   - **Chrome/Edge**: Open `chrome://extensions/`, enable "Developer mode", click "Load unpacked", and select the `extension` folder
   - **Firefox**: Open `about:debugging#/runtime/this-firefox`, click "Load Temporary Add-on", and select any file in the `extension` folder

3. Configure the API endpoint in the extension's popup (default: `http://localhost:8000`)

## 🔌 API Endpoints

### Image Prediction
```
POST /predict/image
Content-Type: multipart/form-data

Body:
- file: Image file (jpeg, png, etc.)

Response:
{
  "is_fake": boolean,
  "confidence": float (0-1),
  "message": string
}
```

### Video Prediction
```
POST /predict/video
Content-Type: multipart/form-data

Body:
- file: Video file (mp4, avi, mov, etc.)

Response:
{
  "is_fake": boolean,
  "confidence": float (0-1),
  "message": string
}
```

## 🔧 Configuration

### Backend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |
| `MODEL_CHECKPOINT_PATH` | `models/deepfake_model.pt` | Path to the ML model |
| `FAKE_THRESHOLD` | `0.5` | Confidence threshold for fake detection |
| `VIDEO_SAMPLE_FRAMES` | `12` | Number of frames to sample from videos |

### Frontend Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes | Backend API base URL |

## 📦 Dependencies

### Backend
- **fastapi**: Web framework for building APIs
- **uvicorn**: ASGI server
- **pydantic**: Data validation
- **torch, torchvision, pillow**: Deep learning and image processing (ML features)
- **opencv-python**: Video processing

### Frontend
- **Next.js**: React framework for production
- **TypeScript**: Type-safe JavaScript
- **ESLint**: Code linting

### Extension
- **Vanilla JavaScript**: No build step required

## 🛠️ Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Quality

```bash
# Frontend linting
cd frontend
npm run lint

# Backend formatting
cd backend
black . && isort .
```

## 📝 API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Backend won't start
- Ensure Python version is 3.10+
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify the port 8000 is not in use: `lsof -i :8000`

### Frontend connection errors
- Confirm backend is running and accessible
- Check `NEXT_PUBLIC_API_URL` matches your backend URL
- Review browser console for CORS errors

### Extension not detecting API
- Ensure backend server is running
- Check the extension's configured API URL matches your backend
- Verify CORS is properly configured in the backend

## 📞 Support

For issues or questions, please open an issue on the repository.

---

**Last Updated**: May 2026
