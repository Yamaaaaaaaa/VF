import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Cell, ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts';

const SEGMENT_COLORS = {
  'MFCC Mean': '#FF5500',
  'MFCC Std':  '#FF8A00',
  'Spec. Contrast': '#6C63FF',
  'Chroma': '#00C9A7',
};

export default function FeatureVectorChart({ embedding }) {
  if (!embedding?.length) return null;

  // Split 99D vector into 4 named segments
  const segments = [
    { label: 'MFCC Mean', values: embedding.slice(0, 40),   color: SEGMENT_COLORS['MFCC Mean'] },
    { label: 'MFCC Std',  values: embedding.slice(40, 80),  color: SEGMENT_COLORS['MFCC Std'] },
    { label: 'Spec. Contrast', values: embedding.slice(80, 87), color: SEGMENT_COLORS['Spec. Contrast'] },
    { label: 'Chroma',    values: embedding.slice(87, 99),  color: SEGMENT_COLORS['Chroma'] },
  ];

  const data = segments.flatMap(({ label, values, color }) =>
    values.map((v, i) => ({ name: `${label[0]}${i}`, value: parseFloat(v.toFixed(3)), segment: label, color }))
  );

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div style={{ background: '#1e1e2e', border: '1px solid #333', borderRadius: 8, padding: '8px 12px', fontSize: '0.8rem' }}>
        <div style={{ color: d.color, fontWeight: 700 }}>{d.segment}</div>
        <div style={{ color: '#fff' }}>Value: <b>{d.value}</b></div>
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 0 }} barSize={4}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey="name" tick={false} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: '#888', fontSize: 11 }} axisLine={false} tickLine={false} width={40} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
        <Bar dataKey="value" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
