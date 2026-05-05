import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Cell, ResponsiveContainer, LabelList,
} from 'recharts';

export default function SimilarityRankingChart({ results }) {
  if (!results?.length) return null;

  const data = results.map((r, i) => ({
    name: `File #${r.file_id}`,
    similarity: parseFloat((r.similarity * 100).toFixed(2)),
    rank: i + 1,
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{ background: '#1e1e2e', border: '1px solid #333', borderRadius: 8, padding: '8px 12px', fontSize: '0.8rem' }}>
        <div style={{ color: '#FF5500', fontWeight: 700 }}>{payload[0].payload.name}</div>
        <div style={{ color: '#fff' }}>Similarity: <b>{payload[0].value}%</b></div>
      </div>
    );
  };

  // Color gradient: rank 1 = brightest orange, rank 5 = muted
  const colors = ['#FF5500', '#FF6B22', '#FF8144', '#FF9766', '#FFAD88'];

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 60, bottom: 4, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={false} />
        <XAxis type="number" domain={[0, 100]} tick={{ fill: '#888', fontSize: 11 }}
          axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
        <YAxis type="category" dataKey="name" width={80}
          tick={{ fill: '#ccc', fontSize: 12 }} axisLine={false} tickLine={false} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
        <Bar dataKey="similarity" radius={[0, 4, 4, 0]} barSize={22}>
          {data.map((_, i) => <Cell key={i} fill={colors[i] || '#FFAD88'} />)}
          <LabelList dataKey="similarity" position="right"
            formatter={v => `${v}%`} style={{ fill: '#aaa', fontSize: 11 }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
