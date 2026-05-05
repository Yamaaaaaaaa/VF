import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { X, Loader } from 'lucide-react';
import WaveformChart from './charts/WaveformChart';
import SpectrogramChart from './charts/SpectrogramChart';
import MFCCChart from './charts/MFCCChart';
import FeatureVectorChart from './charts/FeatureVectorChart';
import RadarComparisonChart from './charts/RadarComparisonChart';
import SimilarityRankingChart from './charts/SimilarityRankingChart';

const API_BASE = 'http://localhost:8000/api';

function ChartCard({ title, subtitle, children, light }) {
  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <span className="chart-card-title">{title}</span>
        {subtitle && <span className="chart-card-subtitle">{subtitle}</span>}
      </div>
      <div className={`chart-card-body${light ? ' light' : ''}` }>{children}</div>
    </div>
  );
}

export default function AnalysisModal({ record, onClose }) {
  const [analysis, setAnalysis] = useState(null);
  const [similar, setSimilar]   = useState([]);
  const [topMatchEmb, setTopMatchEmb] = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);

  useEffect(() => {
    if (!record) return;
    setLoading(true); setError(null);

    Promise.all([
      axios.get(`${API_BASE}/records/${record.file_id}/analyze`),
      axios.get(`${API_BASE}/records/${record.file_id}/similar?top_k=5`),
    ])
      .then(async ([anaRes, simRes]) => {
        setAnalysis(anaRes.data);
        setSimilar(simRes.data);
        // Fetch top match embedding for radar
        if (simRes.data.length > 0) {
          const topId = simRes.data[0].file_id;
          const embRes = await axios.get(`${API_BASE}/records/${topId}/analyze`);
          setTopMatchEmb(embRes.data.embedding);
        }
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [record?.file_id]);

  if (!record) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div>
            <div className="modal-title">Audio Analysis — File #{record.file_id}</div>
            <div className="modal-subtitle">{record.gender} · {record.accent || 'Unknown'} Accent · {record.age ? `${record.age} yrs` : ''}</div>
          </div>
          <button className="modal-close" onClick={onClose}><X size={20} /></button>
        </div>

        {loading && (
          <div className="modal-loading">
            <Loader size={32} className="spin-anim" style={{ animation: 'spin 1s linear infinite', color: '#FF5500' }} />
            <p>Processing audio features…</p>
          </div>
        )}

        {error && (
          <div className="modal-error">⚠ {error}</div>
        )}

        {!loading && !error && analysis && (
          <div className="modal-body">
            {/* Row 1: Waveform full width */}
            <ChartCard title="① Waveform" subtitle="Raw audio signal — 5 seconds at 16 kHz">
              <WaveformChart data={analysis.waveform} />
            </ChartCard>

            {/* Row 2: Mel Spectrogram + MFCC Matrix */}
            <div className="chart-row-2">
              <ChartCard title="② Mel Spectrogram" subtitle="Time–Frequency (64 mel bands)">
                <SpectrogramChart data={analysis.mel_spectrogram} />
              </ChartCard>
              <ChartCard title="③ MFCC Matrix" subtitle="40 coefficients × frames (diverging)">
                <MFCCChart data={analysis.mfcc_matrix} />
              </ChartCard>
            </div>

            {/* Row 3: Feature Vector 99D full width */}
            <ChartCard
              title="④ Feature Vector 99D"
              subtitle="MFCC Mean(40) · MFCC Std(40) · Spectral Contrast(7) · Chroma(12)"
              light
            >
              <FeatureVectorChart embedding={analysis.embedding} />
            </ChartCard>

            {/* Row 4: Radar + Similarity Ranking */}
            <div className="chart-row-2">
              <ChartCard title="⑤ Chroma Radar" subtitle="Query vs Top Match (12 pitch classes)" light>
                <RadarComparisonChart
                  queryEmbedding={analysis.embedding}
                  matchEmbedding={topMatchEmb}
                  matchId={similar[0]?.file_id}
                />
              </ChartCard>
              <ChartCard title="⑥ Similarity Ranking" subtitle="Top 5 cosine similarity scores" light>
                <SimilarityRankingChart results={similar} />
              </ChartCard>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
