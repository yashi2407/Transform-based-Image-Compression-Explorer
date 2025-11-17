export function toGrayscaleFloat(imgEl, B, hiddenCanvas) {
  const w = imgEl.naturalWidth;
  const h = imgEl.naturalHeight;
  const canvas = hiddenCanvas;
  const ctx = canvas.getContext("2d");

  const Wc = Math.floor(w / B) * B;
  const Hc = Math.floor(h / B) * B;

  canvas.width = Wc;
  canvas.height = Hc;
  ctx.drawImage(imgEl, 0, 0, Wc, Hc);

  const imageData = ctx.getImageData(0, 0, Wc, Hc);
  const data = imageData.data;
  const gray = new Float32Array(Wc * Hc);

  for (let i = 0; i < Wc * Hc; i++) {
    const r = data[4 * i] / 255;
    const g = data[4 * i + 1] / 255;
    const b = data[4 * i + 2] / 255;
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  return { width: Wc, height: Hc, gray };
}

export function drawGrayToCanvas(gray, w, h, canvas) {
  if (!canvas) return;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(w, h);
  const data = imageData.data;
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(1, gray[i]));
    const c = Math.round(v * 255);
    data[4 * i] = c;
    data[4 * i + 1] = c;
    data[4 * i + 2] = c;
    data[4 * i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

export function extractBlocksJS(gray, width, height, B) {
  const nrows = height / B;
  const ncols = width / B;
  const blocks = [];
  const d = B * B;
  for (let by = 0; by < nrows; by++) {
    for (let bx = 0; bx < ncols; bx++) {
      const block = new Array(d);
      let idx = 0;
      for (let y = 0; y < B; y++) {
        for (let x = 0; x < B; x++) {
          const gx = bx * B + x;
          const gy = by * B + y;
          block[idx++] = gray[gy * width + gx];
        }
      }
      blocks.push(block);
    }
  }
  return { blocks, shape: { nrows, ncols, B } };
}

export function reconstructFromBlocksJS(blocks, shape, width, height) {
  const { nrows, ncols, B } = shape;
  const gray = new Float32Array(width * height);
  let bi = 0;
  for (let by = 0; by < nrows; by++) {
    for (let bx = 0; bx < ncols; bx++) {
      const block = blocks[bi++];
      let idx = 0;
      for (let y = 0; y < B; y++) {
        for (let x = 0; x < B; x++) {
          const gx = bx * B + x;
          const gy = by * B + y;
          gray[gy * width + gx] = block[idx++];
        }
      }
    }
  }
  return gray;
}
