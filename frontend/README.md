This is a Next.js frontend for the DeepFake Detection full-stack app.

## Full-stack setup 
 
1. Start backend API:

```bash
cd ../backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Note: this mocked backend path supports Python 3.14+. For full ML inference (`requirements-ml.txt`), use Python 3.11 or 3.12.

If you want to run real model inference later (instead of the current mocked scoring), install optional ML deps too:

```bash
pip install -r requirements-ml.txt
```

To enable model-backed inference (higher accuracy), set backend env vars before starting Uvicorn:

```bash
export MODEL_CHECKPOINT_PATH=/absolute/path/to/deepfake_model.pt
export FAKE_THRESHOLD=0.5
export VIDEO_SAMPLE_FRAMES=12
```

Then start backend. The `/health` endpoint will show whether `model_loaded` is true.

2. Configure frontend environment:

```bash
cp .env.example .env.local
```

By default, `NEXT_PUBLIC_API_BASE_URL` is set to `http://localhost:8000`.

3. Start frontend:

```bash
npm install
npm run dev
```

4. Optional extension setup:
- Load the `extension/` folder in Chrome as an unpacked extension.
- Open the extension popup and set `Backend API URL` (for local: `http://localhost:8000`).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
