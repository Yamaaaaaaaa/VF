import React, { useEffect, useRef } from 'react';

// Diverging colormap: negative=blue, zero=white, positive=red
function diverging(t) {
  // t in [0,1], 0.5 = white
  if (t < 0.5) {
    const f = t * 2;
    return [Math.round(50 + f * 205), Math.round(50 + f * 205), 255];
  } else {
    const f = (t - 0.5) * 2;
    return [255, Math.round(255 - f * 205), Math.round(255 - f * 205)];
  }
}

export default function MFCCChart({ data }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!data?.length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rows = data.length, cols = data[0].length;
    canvas.width = cols; canvas.height = rows;

    let min = Infinity, max = -Infinity;
    data.forEach(row => row.forEach(v => { if (v < min) min = v; if (v > max) max = v; }));
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;

    const img = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const t = (data[r][c] / absMax + 1) / 2; // normalize to [0,1]
        const [R, G, B] = diverging(Math.max(0, Math.min(1, t)));
        const idx = (r * cols + c) * 4;
        img.data[idx] = R; img.data[idx+1] = G; img.data[idx+2] = B; img.data[idx+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data]);

  return (
    <canvas ref={canvasRef}
      style={{ width: '100%', height: '160px', borderRadius: '8px', display: 'block', imageRendering: 'pixelated' }} />
  );
}
