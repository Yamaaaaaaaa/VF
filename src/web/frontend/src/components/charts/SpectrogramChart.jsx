import React, { useEffect, useRef } from 'react';

// Viridis-like colormap: maps [0,1] → RGB
function viridis(t) {
  const stops = [
    [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
  ];
  const idx = t * (stops.length - 1);
  const lo = Math.floor(idx), hi = Math.min(lo + 1, stops.length - 1);
  const f = idx - lo;
  return stops[lo].map((c, i) => Math.round(c + f * (stops[hi][i] - c)));
}

export default function SpectrogramChart({ data }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!data?.length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rows = data.length, cols = data[0].length;
    canvas.width = cols; canvas.height = rows;

    // Find global min/max for normalization
    let min = Infinity, max = -Infinity;
    data.forEach(row => row.forEach(v => { if (v < min) min = v; if (v > max) max = v; }));
    const range = max - min || 1;

    const img = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const t = (data[rows - 1 - r][c] - min) / range; // flip Y
        const [R, G, B] = viridis(t);
        const idx = (r * cols + c) * 4;
        img.data[idx] = R; img.data[idx+1] = G; img.data[idx+2] = B; img.data[idx+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data]);

  return (
    <canvas ref={canvasRef}
      style={{ width: '100%', height: '140px', borderRadius: '8px', display: 'block', imageRendering: 'pixelated' }} />
  );
}
