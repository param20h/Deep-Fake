# DeepFake Detector - Frontend

A modern Next.js web application for deepfake detection. Upload images or videos to instantly analyze whether they contain AI-generated or manipulated faces.

## 🚀 Getting Started

### Prerequisites
- Node.js 18 or higher
- npm or yarn package manager

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create a `.env.local` file in the root directory:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Start the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## 📚 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint code quality checks

## 🏗️ Project Structure

```
src/
├── app/
│   ├── layout.tsx       # Root layout component
│   ├── page.tsx         # Home page
│   ├── page.module.css  # Page styles
│   └── globals.css      # Global styles
└── ...
```

## 🔌 API Integration

This frontend connects to the DeepFake Detector API running on `NEXT_PUBLIC_API_URL`. Ensure the backend is running before using the application.

### Example API Call:
```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict/image`, {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result); // { is_fake, confidence, message }
```

## 🎨 Styling

This project uses CSS Modules for component-scoped styling. Global styles are defined in `globals.css`.

## 📖 Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [TypeScript in Next.js](https://nextjs.org/docs/basic-features/typescript)

## 🆘 Troubleshooting

**Connection Error to API?**
- Ensure the backend is running on the configured `NEXT_PUBLIC_API_URL`
- Check CORS settings in the backend configuration
- Verify the URL in `.env.local` is correct

**Port 3000 already in use?**
```bash
npm run dev -- -p 3001
```

---

Part of the [DeepFake Detector](../README.md) project suite.