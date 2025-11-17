// src/lib/transforms.js
import * as numericModule from "numeric";

// local alias + global (for safety)
export const numeric = numericModule;
if (typeof window !== "undefined") {
  window.numeric = numericModule;
}

// ========= Utility =========

export function psnrJS(x, x_hat, maxVal = 1.0) {
  let mse = 0;
  const n = x.length;
  for (let i = 0; i < n; i++) {
    const diff = x[i] - x_hat[i];
    mse += diff * diff;
  }
  mse /= n;
  if (mse === 0) return 99.0;
  return 10 * Math.log10((maxVal * maxVal) / mse);
}

// ======= Transforms =======

export function dctMatrixJS(N) {
  const C = numeric.rep([N, N], 0);
  const alpha = (k) => (k === 0 ? Math.sqrt(1.0 / N) : Math.sqrt(2.0 / N));
  for (let k = 0; k < N; k++) {
    for (let n = 0; n < N; n++) {
      C[k][n] = alpha(k) * Math.cos((Math.PI * (2 * n + 1) * k) / (2.0 * N));
    }
  }
  return C;
}

export function hadamardMatrixJS(N) {
  if ((N & (N - 1)) !== 0) {
    throw new Error("N must be power of 2 for Hadamard");
  }
  function buildHad(n) {
    if (n === 1) return [[1]];
    const Hsmall = buildHad(n / 2);
    const H = numeric.rep([n, n], 0);
    for (let i = 0; i < n / 2; i++) {
      for (let j = 0; j < n / 2; j++) {
        const v = Hsmall[i][j];
        H[i][j] = v;
        H[i][j + n / 2] = v;
        H[i + n / 2][j] = v;
        H[i + n / 2][j + n / 2] = -v;
      }
    }
    return H;
  }
  let H = buildHad(N);
  const scale = 1 / Math.sqrt(N);
  H = numeric.mul(scale, H);
  return H;
}

export function kron2DFrom1DJS(T1d) {
  const N = T1d.length;
  const result = numeric.rep([N * N, N * N], 0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const a = T1d[i][j];
      for (let p = 0; p < N; p++) {
        for (let q = 0; q < N; q++) {
          result[i * N + p][j * N + q] = a * T1d[p][q];
        }
      }
    }
  }
  return result;
}

export function pcaTransformJS(blocksFlat) {
  const X = blocksFlat.map((row) => row.slice());
  const numSamples = X.length;
  const d = X[0].length;

  // 1) Compute mean
  const mean = new Array(d).fill(0);
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < d; j++) {
      mean[j] += X[i][j];
    }
  }
  for (let j = 0; j < d; j++) {
    mean[j] /= numSamples;
  }

  // 2) Centered data
  const Xc = numeric.rep([numSamples, d], 0);
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < d; j++) {
      Xc[i][j] = X[i][j] - mean[j];
    }
  }

  let V;

  if (numSamples >= d) {
    // "Tall" matrix: OK for numeric.svd
    const svd = numeric.svd(Xc);
    V = svd.V; // d × d
  } else {
    // "Fat" matrix: do SVD on transpose instead
    // Xc^T has shape d × numSamples (rows >= cols)
    const XcT = numeric.transpose(Xc);
    const svdT = numeric.svd(XcT);
    // For XcT, we have XcT = U Σ V^T, and U (d × d) contains
    // the principal directions in the original feature space.
    V = svdT.U; // d × d
  }

  const T = numeric.transpose(V); // rows of T are basis vectors
  return { T, mean };
}

// ======= Transform / inverse per block =======

export function transformBlocksJS(blocksFlat, T, meanVec = null) {
  const numBlocks = blocksFlat.length;
  const d = blocksFlat[0].length;
  const m = T.length;
  const coeffs = numeric.rep([numBlocks, m], 0);

  const TT = numeric.transpose(T);
  for (let i = 0; i < numBlocks; i++) {
    const x = blocksFlat[i].slice();
    if (meanVec) {
      for (let j = 0; j < d; j++) x[j] -= meanVec[j];
    }
    const y = numeric.dot(x, TT);
    for (let j = 0; j < m; j++) coeffs[i][j] = y[j];
  }
  return coeffs;
}

export function inverseTransformBlocksJS(coeffs, T, meanVec = null) {
  const numBlocks = coeffs.length;
  const d = T[0].length;
  const blocks = new Array(numBlocks);
  for (let i = 0; i < numBlocks; i++) {
    const y = coeffs[i];
    const x = numeric.dot(y, T);
    if (meanVec) {
      for (let j = 0; j < d; j++) x[j] += meanVec[j];
    }
    blocks[i] = x;
  }
  return blocks;
}

// ======= Compression: keep top-k =======

export function keepTopK(coeffs, k) {
  const numBlocks = coeffs.length;
  const m = coeffs[0].length;
  const Y = coeffs.map((row) => row.slice());
  if (k >= m) return Y;

  for (let i = 0; i < numBlocks; i++) {
    const row = Y[i];
    const mags = row.map(Math.abs);
    const sorted = mags.slice().sort((a, b) => b - a); // descending
    const thresh = sorted[k - 1]; // k-th largest
    for (let j = 0; j < m; j++) {
      if (
        Math.abs(row[j]) < thresh ||
        (j >= k && Math.abs(row[j]) === thresh)
      ) {
        row[j] = 0;
      }
    }
  }
  return Y;
}

// ======= Energy compaction =======

export function energyCompactionCurveJS(coeffs, maxK) {
  const numBlocks = coeffs.length;
  const m = coeffs[0].length;
  if (!maxK) maxK = m;
  const ks = [];
  const avgFrac = [];

  const sortedSq = [];
  const totalEnergy = new Array(numBlocks);

  for (let i = 0; i < numBlocks; i++) {
    const sq = coeffs[i].map((v) => v * v);
    sq.sort((a, b) => b - a);
    sortedSq[i] = sq;
    totalEnergy[i] = sq.reduce((acc, v) => acc + v, 0) + 1e-12;
  }

  for (let k = 1; k <= maxK; k++) {
    let sumFrac = 0;
    for (let i = 0; i < numBlocks; i++) {
      let partial = 0;
      for (let j = 0; j < k; j++) partial += sortedSq[i][j];
      sumFrac += partial / totalEnergy[i];
    }
    ks.push(k);
    avgFrac.push(sumFrac / numBlocks);
  }

  return { ks, avgFrac };
}

// ======= Rate–distortion =======

export function rateDistortionJS(
  imgGray,
  width,
  height,
  blocksFlat,
  shape,
  T,
  meanVec,
  kValues
) {
  const d = blocksFlat[0].length;
  const coeffs = transformBlocksJS(blocksFlat, T, meanVec);
  const rates = [];
  const psnrs = [];

  for (const k of kValues) {
    const Yk = keepTopK(coeffs, k);
    const blocksRec = inverseTransformBlocksJS(Yk, T, meanVec);
    const grayRec = reconstructFromBlocksJS(blocksRec, shape, width, height);
    for (let i = 0; i < grayRec.length; i++) {
      if (grayRec[i] < 0) grayRec[i] = 0;
      if (grayRec[i] > 1) grayRec[i] = 1;
    }
    const P = psnrJS(imgGray, grayRec);
    rates.push(k / d);
    psnrs.push(P);
  }

  return { rates, psnrs };
}
// We need reconstructFromBlocksJS here; either import from imageUtils,
// or if you prefer keep RD stuff in App.jsx and just export transform functions.
import { reconstructFromBlocksJS } from "./imageUtils";
