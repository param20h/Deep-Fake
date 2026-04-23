# DeepFake Detection System - Next Level App Implementation Plan

We will build a "next level", premium web application for the DeepFake Detection System, improving upon the basic Gradio interface suggested in the blueprint to deliver a truly impressive portfolio piece. The system will consist of a modern frontend, a robust API backend, and a Chrome extension, all designed to be deployed 100% for free.

## User Review Required

> [!IMPORTANT]  
> **Frontend Architecture Choice**: The blueprint suggests Gradio for the UI. To achieve the "next level" premium look you requested, I propose we build a custom **Next.js** frontend deployed on Vercel (free), paired with a **FastAPI** backend deployed on Render or HuggingFace Spaces (free). Does this sound good, or would you prefer to stick strictly to Gradio?
>
> **Styling**: I will use Vanilla CSS with modern aesthetics (glassmorphism, dark mode, smooth animations) as per best practices. Let me know if you strongly prefer a framework like TailwindCSS.

## Open Questions

> [!NOTE]  
> 1. Do you already have a trained PyTorch/EfficientNet model (`.pt` file), or should I create the backend with a placeholder prediction function for now so we can focus on the full-stack plumbing first?
> 2. Should we initialize the Next.js app and FastAPI backend in the current directory (`/Users/param/study/CVDL/deepfake`)?

## Proposed Architecture & Changes

The project will be split into three main components:

### 1. Next.js Frontend (Vercel - Free Tier)
A highly polished, responsive web application for users to upload images/videos and view analysis.
- **Features**: Drag-and-drop upload, dynamic confidence meters, frame-by-frame timeline for videos, and Grad-CAM heatmap visualization placeholders.
- **Design**: Premium dark-mode aesthetic with vibrant accent colors (e.g., danger red for "Fake", success green for "Real"), micro-animations, and custom typography.
- **Tools**: Next.js, Vanilla CSS.

### 2. FastAPI Backend (Render / HuggingFace Spaces - Free Tier)
The core processing server that handles inference.
- **Features**: Endpoints for `/predict/image` and `/predict/video`. Handles CORS properly so the web app and Chrome extension can communicate with it.
- **Tools**: FastAPI, Uvicorn, PyTorch (mocked initially if model is unavailable), OpenCV.

### 3. Chrome Extension (Manifest V3)
A browser extension to check images on the fly.
- **Features**: Right-click context menu ("Check if deepfake"), which sends the image URL to the FastAPI backend and displays a score badge.
- **Tools**: Vanilla JS, HTML/CSS.

## Verification Plan

### Automated Tests
- Validate FastAPI endpoints using built-in Swagger UI (`/docs`).
- Test Next.js UI components for responsiveness across desktop and mobile views.

### Manual Verification
- Run the Next.js dev server and verify the UI aesthetics and upload flows.
- Run the FastAPI dev server and verify successful mock predictions.
- Load the unpacked Chrome extension locally and test the right-click workflow on a public image.
