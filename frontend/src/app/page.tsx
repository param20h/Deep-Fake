'use client';

import { useState, useRef } from 'react';
import Image from 'next/image';
import styles from './page.module.css';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<{is_fake: boolean, confidence: number} | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (selectedFile: File) => {
    setFile(selectedFile);
    setResult(null);
    setErrorMessage(null);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
  };

  const analyzeFile = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setResult(null);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append('file', file);

    const endpoint = file.type.startsWith('video/') 
      ? `${API_BASE_URL}/predict/video`
      : `${API_BASE_URL}/predict/image`;

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || 'Analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      setErrorMessage('Failed to analyze file. Check backend URL and ensure API is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="shell">
      <div className="scanline" />

      <section className={`${styles.hero} fade-up`}>
        <p className={styles.kicker}>AI Forensics Lab</p>
        <h1>
          Spot manipulated media with a
          <span> clear confidence verdict </span>
          in seconds.
        </h1>
        <p className={styles.subtitle}>
          Upload an image or a short video. The backend runs deepfake scoring and returns a confidence-driven result for fast triage.
        </p>

        <div className={styles.metricRow}>
          <div className={styles.metric}>
            <span>Mode</span>
            <strong>Image + Video</strong>
          </div>
          <div className={styles.metric}>
            <span>Latency</span>
            <strong>Realtime API</strong>
          </div>
          <div className={styles.metric}>
            <span>System</span>
            <strong>Connected</strong>
          </div>
        </div>
      </section>

      <section className={`${styles.workspace} panel fade-up`}>
        <div className={styles.workspaceHeader}>
          <h2>Detection Workspace</h2>
          <span className={styles.statusDot}>{isAnalyzing ? 'Analyzing' : 'Ready'}</span>
        </div>

        {!file ? (
          <div
            className={styles.dropzone}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                fileInputRef.current?.click();
              }
            }}
          >
            <div className={styles.dropIcon}>▲</div>
            <h3>Drop media here or click to browse</h3>
            <p>Accepted: JPG, PNG, MP4. Best results with clear frontal faces.</p>
            <input
              type="file"
              ref={fileInputRef}
              className={styles.hiddenInput}
              accept="image/*,video/*"
              onChange={handleFileChange}
            />
          </div>
        ) : (
          <div className={styles.loadedState}>
            <div className={styles.fileRow}>
              <div>
                <h3>{file.name}</h3>
                <p>{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
              <button
                className={styles.ghostBtn}
                onClick={() => {
                  setFile(null);
                  setPreviewUrl(null);
                  setResult(null);
                  setErrorMessage(null);
                }}
              >
                Replace File
              </button>
            </div>

            {previewUrl && (
              <div className={styles.previewBox}>
                {file.type.startsWith('video/') ? (
                  <video src={previewUrl} controls className={styles.previewMedia} />
                ) : (
                  <Image
                    src={previewUrl}
                    alt="Uploaded preview"
                    className={styles.previewMedia}
                    width={1200}
                    height={780}
                    unoptimized
                  />
                )}
              </div>
            )}

            {!result && (
              <>
                <button
                  className={styles.primaryBtn}
                  onClick={analyzeFile}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? 'Running Deep Analysis...' : 'Run Deepfake Detection'}
                </button>
                {errorMessage && <p className={styles.errorText}>{errorMessage}</p>}
              </>
            )}

            {result && (
              <div className={styles.resultCard}>
                <div className={styles.resultHead}>
                  <h3>Analysis Complete</h3>
                  <span className={`${styles.verdict} ${result.is_fake ? styles.fake : styles.real}`}>
                    {result.is_fake ? 'Fake Likely' : 'Likely Authentic'}
                  </span>
                </div>

                <div className={styles.progressWrap}>
                  <div className={styles.progressMeta}>
                    <span>Confidence</span>
                    <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                  </div>
                  <div className={styles.track}>
                    <div
                      className={`${styles.fill} ${result.is_fake ? styles.fillFake : styles.fillReal}`}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>

                <p className={styles.resultNote}>
                  {file.type.startsWith('video/')
                    ? 'Video sampled across multiple frames. Temporal consistency contributed to this score.'
                    : 'Image-level artifact patterns and facial integrity signals contributed to this score.'}
                </p>
              </div>
            )}
          </div>
        )}
      </section>

      <footer className={`${styles.footer} fade-up`}>
        <div>
          <p className={styles.footerBrand}>™ 2026 DeepFake by Param20h</p>
          <p className={styles.footerCopy}>Built for fast screening, model-backed review, and future training upgrades.</p>
        </div>
        <a
          className={styles.footerLink}
          href="https://github.com/param20h"
          target="_blank"
          rel="noreferrer"
        >
          GitHub / param20h
        </a>
      </footer>
    </main>
  );
}
