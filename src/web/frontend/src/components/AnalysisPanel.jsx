import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Loader } from 'lucide-react';
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
      <div className={`chart-card-body${light ? ' light' : ''}`}>{children}</div>
    </div>
  );
}

export default function AnalysisPanel({ record }) {
  const [analysis, setAnalysis]     = useState(null);
  const [similar, setSimilar]       = useState([]);
  const [topMatchEmb, setTopMatchEmb] = useState(null);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState(null);

  useEffect(() => {
    if (!record) return;
    setLoading(true);
    setError(null);
    setAnalysis(null);
    setSimilar([]);
    setTopMatchEmb(null);

    Promise.all([
      axios.get(`${API_BASE}/records/${record.file_id}/analyze`),
      axios.get(`${API_BASE}/records/${record.file_id}/similar?top_k=5`),
    ])
      .then(async ([anaRes, simRes]) => {
        setAnalysis(anaRes.data);
        setSimilar(simRes.data);
        if (simRes.data.length > 0) {
          const embRes = await axios.get(`${API_BASE}/records/${simRes.data[0].file_id}/analyze`);
          setTopMatchEmb(embRes.data.embedding);
        }
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [record?.file_id]);

  return (
    <div className="analysis-panel fade-in">
      {/* Panel title bar */}
      <div className="analysis-panel-header">
        <div className="analysis-panel-title">
          Audio Analysis — <span style={{ color: 'var(--primary-orange)' }}>File #{record.file_id}</span>
        </div>
        <div className="analysis-panel-meta">
          {record.gender} · {record.accent || 'Unknown'} Accent{record.age ? ` · ${record.age} yrs` : ''}
        </div>
      </div>

      {loading && (
        <div className="analysis-panel-loading">
          <Loader size={28} style={{ animation: 'spin 1s linear infinite', color: 'var(--primary-orange)' }} />
          <span>Processing audio features…</span>
        </div>
      )}

      {error && <div className="analysis-panel-error">⚠ {error}</div>}

      {!loading && !error && analysis && (
        <div className="analysis-charts-grid">
          {/* Row 1 — Waveform (full width) */}
          <div className="chart-full">
            <ChartCard title="① Waveform" subtitle="Raw audio signal · 5s @ 16 kHz">
              <WaveformChart data={analysis.waveform} />
            </ChartCard>
          </div>

          {/* Row 2 — Mel Spectrogram + MFCC */}
          <div className="chart-half">
            <ChartCard title="② Mel Spectrogram" subtitle="Time–Frequency · 64 mel bands">
              <SpectrogramChart data={analysis.mel_spectrogram} />
            </ChartCard>
          </div>
          <div className="chart-half">
            <ChartCard title="③ MFCC Matrix" subtitle="40 coefficients × frames">
              <MFCCChart data={analysis.mfcc_matrix} />
            </ChartCard>
          </div>

          {/* Row 3 — Feature Vector (full width) */}
          <div className="chart-full">
            <ChartCard
              title="④ Feature Vector 99D"
              subtitle="MFCC Mean(40) · MFCC Std(40) · Spectral Contrast(7) · Chroma(12)"
              light
            >
              <FeatureVectorChart embedding={analysis.embedding} />
            </ChartCard>
          </div>

          {/* Row 4 — Radar + Ranking */}
          <div className="chart-half">
            <ChartCard title="⑤ Chroma Radar" subtitle="Selected vs Top Match · 12 pitch classes" light>
              <RadarComparisonChart
                queryEmbedding={analysis.embedding}
                matchEmbedding={topMatchEmb}
                matchId={similar[0]?.file_id}
              />
            </ChartCard>
          </div>
          <div className="chart-half">
            <ChartCard title="⑥ Similarity Ranking" subtitle="Top 5 cosine similarity scores" light>
              <SimilarityRankingChart results={similar} />
            </ChartCard>
          </div>
        </div>
      )}
    </div>
  );
}
