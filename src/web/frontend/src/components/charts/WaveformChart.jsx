import React, { useEffect, useRef } from 'react';

export default function WaveformChart({ data }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!data?.length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#12121a';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach(r => {
      ctx.beginPath(); ctx.moveTo(0, H * r); ctx.lineTo(W, H * r); ctx.stroke();
    });

    // Waveform path
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, '#FF5500');
    grad.addColorStop(0.5, '#FF8A00');
    grad.addColorStop(1, '#FF5500');

    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * W;
      const y = H / 2 - (v * H * 0.45);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = grad;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Fill under curve
    ctx.lineTo(W, H / 2);
    ctx.lineTo(0, H / 2);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,85,0,0.12)';
    ctx.fill();
  }, [data]);

  return (
    <canvas ref={canvasRef} width={900} height={120}
      style={{ width: '100%', height: '120px', borderRadius: '8px', display: 'block' }} />
  );
}
