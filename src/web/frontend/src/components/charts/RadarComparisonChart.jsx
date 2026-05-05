import React from 'react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer, Legend, Tooltip,
} from 'recharts';

const CHROMA_LABELS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

export default function RadarComparisonChart({ queryEmbedding, matchEmbedding, matchId }) {
  if (!queryEmbedding?.length || !matchEmbedding?.length) return null;

  // Use the Chroma slice (indices 87-99) — 12 pitch classes
  const queryChroma = queryEmbedding.slice(87, 99);
  const matchChroma = matchEmbedding.slice(87, 99);

  const data = CHROMA_LABELS.map((label, i) => ({
    axis: label,
    query: parseFloat(queryChroma[i]?.toFixed(3) ?? 0),
    match: parseFloat(matchChroma[i]?.toFixed(3) ?? 0),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
        <PolarGrid stroke="rgba(255,255,255,0.1)" />
        <PolarAngleAxis dataKey="axis" tick={{ fill: '#aaa', fontSize: 12 }} />
        <PolarRadiusAxis angle={90} tick={{ fill: '#666', fontSize: 10 }} />
        <Tooltip
          contentStyle={{ background: '#1e1e2e', border: '1px solid #333', borderRadius: 8, fontSize: '0.8rem' }}
          labelStyle={{ color: '#fff' }}
        />
        <Legend wrapperStyle={{ fontSize: '0.8rem' }} />
        <Radar name="Selected" dataKey="query"
          stroke="#FF5500" fill="#FF5500" fillOpacity={0.25} strokeWidth={2} dot />
        <Radar name={`File #${matchId} (Top Match)`} dataKey="match"
          stroke="#6C63FF" fill="#6C63FF" fillOpacity={0.2} strokeWidth={2} dot />
      </RadarChart>
    </ResponsiveContainer>
  );
}
