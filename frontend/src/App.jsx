import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Microscope, Upload, Activity, AlertTriangle } from 'lucide-react';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const topPrediction = useMemo(() => result?.top3?.[0], [result]);

  function selectFile(nextFile) {
    setFile(nextFile);
    setResult(null);
    setError('');
    setPreview(nextFile ? URL.createObjectURL(nextFile) : '');
  }

  async function classify() {
    if (!file) return;
    setBusy(true);
    setError('');
    const form = new FormData();
    form.append('file', file);
    try {
      const response = await fetch(`${API_BASE}/classify`, { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Classification failed');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <section className="workspace">
        <div className="topbar">
          <div className="brand">
            <Microscope size={28} />
            <div>
              <h1>BiT Bacterial Classifier</h1>
              <p>Microscopy image classification with confidence scores and Grad-CAM preview</p>
            </div>
          </div>
          <span className="mode">{result?.mode || 'demo'} mode</span>
        </div>

        <div className="grid">
          <section className="panel upload-panel">
            <label
              className="dropzone"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                selectFile(event.dataTransfer.files?.[0]);
              }}
            >
              <input type="file" accept="image/*" onChange={(event) => selectFile(event.target.files?.[0])} />
              {preview ? <img src={preview} alt="Uploaded microscopy preview" /> : <Upload size={42} />}
            </label>
            <button className="primary" disabled={!file || busy} onClick={classify}>
              <Activity size={18} />
              {busy ? 'Classifying...' : 'Classify Image'}
            </button>
            {error && <div className="error"><AlertTriangle size={16} />{error}</div>}
          </section>

          <section className="panel">
            <h2>Predictions</h2>
            {!result && <p className="empty">Upload a microscopy image to view ranked bacterial species predictions.</p>}
            {result?.top3?.map((item, index) => (
              <div className="prediction" key={item.class_id}>
                <div className="prediction-row">
                  <strong>{index + 1}. {item.species}</strong>
                  <span>{Math.round(item.confidence * 100)}%</span>
                </div>
                <div className="bar"><span style={{ width: `${item.confidence * 100}%` }} /></div>
              </div>
            ))}
          </section>

          <section className="panel heatmap-panel">
            <h2>Grad-CAM</h2>
            {result?.heatmap_url ? <img src={result.heatmap_url} alt="Grad-CAM heatmap overlay" /> : <p className="empty">The heatmap appears after classification.</p>}
          </section>

          <section className="panel">
            <h2>Clinical Metadata</h2>
            {topPrediction && result?.clinical_info ? (
              <div className="metadata">
                <h3>{topPrediction.species}</h3>
                <dl>
                  <dt>Gram type</dt><dd>{result.clinical_info.gram_type}</dd>
                  <dt>Genus</dt><dd>{result.clinical_info.genus}</dd>
                  <dt>Sites</dt><dd>{result.clinical_info.infection_sites.join(', ')}</dd>
                  <dt>Antibiotics</dt><dd>{result.clinical_info.antibiotics.join(', ')}</dd>
                  <dt>Focus score</dt><dd>{result.focus_score}</dd>
                  <dt>Latency</dt><dd>{result.inference_time_ms} ms</dd>
                </dl>
                <p>{result.clinical_info.notes}</p>
              </div>
            ) : <p className="empty">Species metadata is shown with the top prediction.</p>}
          </section>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
