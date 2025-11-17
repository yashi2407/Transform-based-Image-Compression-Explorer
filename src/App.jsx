import React, { useState, useRef, useCallback } from "react";
import {
  Chart,
  LineElement,
  PointElement,
  LineController,
  CategoryScale,
  LinearScale,
  Legend,
  Tooltip,
} from "chart.js";

import "./App.css";
import * as numericModule from "numeric";

Chart.register(
  LineElement,
  PointElement,
  LineController,
  CategoryScale,
  LinearScale,
  Legend,
  Tooltip
);

import {
  toGrayscaleFloat,
  drawGrayToCanvas,
  extractBlocksJS,
  reconstructFromBlocksJS,
} from "./lib/imageUtils";

import {
  psnrJS,
  dctMatrixJS,
  hadamardMatrixJS,
  kron2DFrom1DJS,
  pcaTransformJS,
  transformBlocksJS,
  inverseTransformBlocksJS,
  keepTopK,
  energyCompactionCurveJS,
  rateDistortionJS,
} from "./lib/transforms";

// Create a local alias AND a global one
const numeric = numericModule;

if (typeof window !== "undefined") {
  window.numeric = numericModule;
}

async function runSweepForCurrentImage({
  imgEl,
  hiddenCanvas,
  blockSizes = [4, 8, 16, 32],
  kFractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
}) {
  const results = [];

  for (const B of blockSizes) {
    const imgData = toGrayscaleFloat(imgEl, B, hiddenCanvas);
    const { width, height, gray } = imgData;
    const { blocks, shape } = extractBlocksJS(gray, width, height, B);
    const d = B * B;

    const C1d = dctMatrixJS(B);
    const T_dct = kron2DFrom1DJS(C1d);

    const H1d = hadamardMatrixJS(B);
    const T_had = kron2DFrom1DJS(H1d);

    const { T: T_pca, mean: pcaMean } = pcaTransformJS(blocks);
    const kValues = kFractions.map((f) => Math.max(1, Math.round(f * d)));

    const rdDCT = rateDistortionJS(
      gray,
      width,
      height,
      blocks,
      shape,
      T_dct,
      null,
      kValues
    );
    const rdHAD = rateDistortionJS(
      gray,
      width,
      height,
      blocks,
      shape,
      T_had,
      null,
      kValues
    );
    const rdPCA = rateDistortionJS(
      gray,
      width,
      height,
      blocks,
      shape,
      T_pca,
      pcaMean,
      kValues
    );

    for (let i = 0; i < kValues.length; i++) {
      const k = kValues[i];
      const rate = k / d;

      results.push({
        blockSize: B,
        k,
        rate,
        transform: "DCT",
        psnr: rdDCT.psnrs[i],
      });
      results.push({
        blockSize: B,
        k,
        rate,
        transform: "Hadamard",
        psnr: rdHAD.psnrs[i],
      });
      results.push({
        blockSize: B,
        k,
        rate,
        transform: "PCA",
        psnr: rdPCA.psnrs[i],
      });
    }

    await new Promise((res) => setTimeout(res, 0));
  }

  return results;
}

// ========= React Component =========

export default function App() {
  const [imageUrl, setImageUrl] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [blockSize, setBlockSize] = useState(8);
  const [kShow, setKShow] = useState(16);
  const [status, setStatus] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSweepRunning, setIsSweepRunning] = useState(false);
  const [canRun, setCanRun] = useState(false);
  const [dctMatrix, setDctMatrix] = useState(null);
  const [hadMatrix, setHadMatrix] = useState(null);

  const originalCanvasRef = useRef(null);
  const dctCanvasRef = useRef(null);
  const hadCanvasRef = useRef(null);
  const pcaCanvasRef = useRef(null);
  const pcaBasisCanvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const energyCanvasRef = useRef(null);
  const rdCanvasRef = useRef(null);

  const energyChartRef = useRef(null);
  const rdChartRef = useRef(null);

  const originalImageRef = useRef(null);
  const blocksFlatRef = useRef(null);
  const blocksShapeRef = useRef(null);
  const T_dctRef = useRef(null);
  const T_hadRef = useRef(null);
  const T_pcaRef = useRef(null);
  const pcaMeanRef = useRef(null);
  const imgElementRef = useRef(null);

  const handleImageLoaded = useCallback(
    (imgEl) => {
      const B = parseInt(blockSize, 10) || 8;
      if (!hiddenCanvasRef.current) return;

      setStatus("Converting to grayscale and preparing data...");
      const imgData = toGrayscaleFloat(imgEl, B, hiddenCanvasRef.current);
      originalImageRef.current = imgData;

      const { gray, width, height } = imgData;
      drawGrayToCanvas(gray, width, height, originalCanvasRef.current);

      const { blocks, shape } = extractBlocksJS(gray, width, height, B);
      blocksFlatRef.current = blocks;
      blocksShapeRef.current = shape;

      setCanRun(true);
      setStatus("Image loaded. Click 'Run Transforms & Analysis'.");
    },
    [blockSize]
  );

  const handleLoadImageClick = useCallback(() => {
    if (!imageUrl && !imageFile) {
      setStatus("Provide an image URL or upload a file.");
      return;
    }

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgElementRef.current = img;
      handleImageLoaded(img);
    };
    img.onerror = () => setStatus("Failed to load image. Check URL or file.");

    if (imageUrl) {
      img.src = imageUrl; // URL has priority
    } else if (imageFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        img.src = e.target.result;
      };
      reader.readAsDataURL(imageFile);
    }
  }, [imageUrl, imageFile, handleImageLoaded]);

  const runAnalysis = useCallback(async () => {
    const originalImage = originalImageRef.current;
    const blocksFlat = blocksFlatRef.current;
    const blocksShape = blocksShapeRef.current;

    if (!originalImage || !blocksFlat || !blocksShape) {
      setStatus("Load an image first.");
      return;
    }

    const B = parseInt(blockSize, 10) || 8;
    const k = parseInt(kShow, 10) || 16;
    const { width, height, gray: imgGray } = originalImage;
    const d = B * B;

    setIsRunning(true);
    setMetrics(null);
    setStatus(
      "Building transforms and running analysis (this may take a few seconds)..."
    );

    // Let React render the loading state before blocking computations
    await new Promise((resolve) => setTimeout(resolve, 0));

    try {
      // Build transforms
      const C1d = dctMatrixJS(B);
      const T_dct = kron2DFrom1DJS(C1d);
      const H1d = hadamardMatrixJS(B);
      const T_had = kron2DFrom1DJS(H1d);
      const { T: T_pca, mean: pcaMean } = pcaTransformJS(blocksFlat);

      T_dctRef.current = T_dct;
      T_hadRef.current = T_had;
      T_pcaRef.current = T_pca;
      pcaMeanRef.current = pcaMean;

      setDctMatrix(C1d);
      setHadMatrix(H1d);

      // PCA first basis vector visualized as N×N
      const firstBasis = T_pca[0];
      const basisGray = new Float32Array(d);
      let minVal = Infinity,
        maxVal = -Infinity;
      for (let i = 0; i < d; i++) {
        minVal = Math.min(minVal, firstBasis[i]);
        maxVal = Math.max(maxVal, firstBasis[i]);
      }
      const range = maxVal - minVal || 1;
      for (let i = 0; i < d; i++) {
        basisGray[i] = (firstBasis[i] - minVal) / range;
      }
      drawGrayToCanvas(basisGray, B, B, pcaBasisCanvasRef.current);

      // Coefficients
      const coeffsDCT = transformBlocksJS(blocksFlat, T_dct, null);
      const coeffsHAD = transformBlocksJS(blocksFlat, T_had, null);
      const coeffsPCA = transformBlocksJS(blocksFlat, T_pca, pcaMean);

      // Energy compaction
      const maxK = d;
      const ecDCT = energyCompactionCurveJS(coeffsDCT, maxK);
      const ecHAD = energyCompactionCurveJS(coeffsHAD, maxK);
      const ecPCA = energyCompactionCurveJS(coeffsPCA, maxK);

      const fracK = ecDCT.ks.map((kk) => kk / d);
      const energyData = {
        labels: fracK,
        datasets: [
          {
            label: "DCT",
            data: ecDCT.avgFrac,
            borderColor: "#22c55e",
            tension: 0.2,
          },
          {
            label: "Hadamard",
            data: ecHAD.avgFrac,
            borderColor: "#3b82f6",
            tension: 0.2,
          },
          {
            label: "PCA (learned)",
            data: ecPCA.avgFrac,
            borderColor: "#f97316",
            tension: 0.2,
          },
        ],
      };

      if (energyChartRef.current) {
        energyChartRef.current.destroy();
      }
      energyChartRef.current = new Chart(
        energyCanvasRef.current.getContext("2d"),
        {
          type: "line",
          data: energyData,
          options: {
            responsive: true,
            plugins: {
              legend: { labels: { color: "#e5e7eb" } },
            },
            scales: {
              x: {
                title: {
                  display: true,
                  text: "k / d",
                  color: "#9ca3af",
                },
                ticks: { color: "#9ca3af" },
                grid: { color: "#1f2937" },
              },
              y: {
                title: {
                  display: true,
                  text: "Average energy fraction",
                  color: "#9ca3af",
                },
                min: 0,
                max: 1,
                ticks: { color: "#9ca3af" },
                grid: { color: "#1f2937" },
              },
            },
          },
        }
      );

      // Rate–distortion
      const kValues = [2, 4, 8, 16, 24, 32, 40, 48, d];
      const rdDCT = rateDistortionJS(
        imgGray,
        width,
        height,
        blocksFlat,
        blocksShape,
        T_dct,
        null,
        kValues
      );
      const rdHAD = rateDistortionJS(
        imgGray,
        width,
        height,
        blocksFlat,
        blocksShape,
        T_had,
        null,
        kValues
      );
      const rdPCA = rateDistortionJS(
        imgGray,
        width,
        height,
        blocksFlat,
        blocksShape,
        T_pca,
        pcaMean,
        kValues
      );

      const rdData = {
        labels: rdDCT.rates,
        datasets: [
          {
            label: "DCT",
            data: rdDCT.psnrs,
            borderColor: "#22c55e",
            tension: 0.2,
          },
          {
            label: "Hadamard",
            data: rdHAD.psnrs,
            borderColor: "#3b82f6",
            tension: 0.2,
          },
          {
            label: "PCA (learned)",
            data: rdPCA.psnrs,
            borderColor: "#f97316",
            tension: 0.2,
          },
        ],
      };

      if (rdChartRef.current) {
        rdChartRef.current.destroy();
      }
      rdChartRef.current = new Chart(rdCanvasRef.current.getContext("2d"), {
        type: "line",
        data: rdData,
        options: {
          responsive: true,
          plugins: {
            legend: { labels: { color: "#e5e7eb" } },
          },
          scales: {
            x: {
              title: {
                display: true,
                text: "Fraction of kept coefficients per block (rate)",
                color: "#9ca3af",
              },
              ticks: { color: "#9ca3af" },
              grid: { color: "#1f2937" },
            },
            y: {
              title: { display: true, text: "PSNR (dB)", color: "#9ca3af" },
              ticks: { color: "#9ca3af" },
              grid: { color: "#1f2937" },
            },
          },
        },
      });

      // Reconstructions at kShow
      const coeffsDCT_k = keepTopK(coeffsDCT, k);
      const blocksDCT_rec = inverseTransformBlocksJS(coeffsDCT_k, T_dct, null);
      const grayDCT = reconstructFromBlocksJS(
        blocksDCT_rec,
        blocksShape,
        width,
        height
      );
      for (let i = 0; i < grayDCT.length; i++) {
        if (grayDCT[i] < 0) grayDCT[i] = 0;
        if (grayDCT[i] > 1) grayDCT[i] = 1;
      }
      drawGrayToCanvas(grayDCT, width, height, dctCanvasRef.current);

      const coeffsHAD_k = keepTopK(coeffsHAD, k);
      const blocksHAD_rec = inverseTransformBlocksJS(coeffsHAD_k, T_had, null);
      const grayHAD = reconstructFromBlocksJS(
        blocksHAD_rec,
        blocksShape,
        width,
        height
      );
      for (let i = 0; i < grayHAD.length; i++) {
        if (grayHAD[i] < 0) grayHAD[i] = 0;
        if (grayHAD[i] > 1) grayHAD[i] = 1;
      }
      drawGrayToCanvas(grayHAD, width, height, hadCanvasRef.current);

      const coeffsPCA_k = keepTopK(coeffsPCA, k);
      const blocksPCA_rec = inverseTransformBlocksJS(
        coeffsPCA_k,
        T_pca,
        pcaMean
      );
      const grayPCA = reconstructFromBlocksJS(
        blocksPCA_rec,
        blocksShape,
        width,
        height
      );
      for (let i = 0; i < grayPCA.length; i++) {
        if (grayPCA[i] < 0) grayPCA[i] = 0;
        if (grayPCA[i] > 1) grayPCA[i] = 1;
      }
      drawGrayToCanvas(grayPCA, width, height, pcaCanvasRef.current);

      const P_dct = psnrJS(imgGray, grayDCT);
      const P_had = psnrJS(imgGray, grayHAD);
      const P_pca = psnrJS(imgGray, grayPCA);

      setMetrics({
        kShow: k,
        d,
        P_dct,
        P_had,
        P_pca,
      });

      setStatus("Done. Explore the plots and reconstructions above.");
    } catch (e) {
      console.error(e);
      setStatus("Error during analysis: " + e.message);
    } finally {
      setIsRunning(false);
    }
  }, [blockSize, kShow]);

  return (
    <div>
      <header>
        <div>
          <div>
            <h1>Transform-based Compression Explorer</h1>
            <span>
              Compare DCT, Hadamard & PCA (Learned) for N×N image blocks
            </span>
          </div>
          <div>
            <span className="tag">Data Compression Term Project</span>
          </div>
        </div>
      </header>

      <main>
        {/* 1. Load image */}
        <section className="card">
          <h2>
            {" "}
            <span className="blue-button">1.</span>
            Load an image
          </h2>
          <div className="controls">
            <div className="controls-group">
              <label>Image URL</label>
              <input
                type="text"
                value={imageUrl}
                onChange={(e) => {
                  setImageUrl(e.target.value);
                  // Optional: if we start using URL, forget the old file
                  setImageFile(null);
                }}
                placeholder="https://example.com/image.jpg"
              />
              <span className="small-text">
                Note: some URLs may be blocked by CORS.
              </span>
              <span className="small-text">
                Please use CORS safe images like - https://picsum.photos/512
              </span>
            </div>
            <div className="controls-group">
              <label>Block size (N × N)</label>
              <input
                type="number"
                min="4"
                max="64"
                value={blockSize}
                onChange={(e) => setBlockSize(e.target.value)}
              />
              <span className="small-text">
                Must be a power of 2 for Hadamard.
              </span>
            </div>
            <div className="controls-group">
              <label>k for visual reconstructions</label>
              <input
                type="number"
                min="1"
                max="64"
                value={kShow}
                onChange={(e) => setKShow(e.target.value)}
              />
              <span className="small-text">
                Number of kept coefficients per block.
              </span>
            </div>
          </div>
          <div className="upload-wrapper">
            {/* Divider line with label */}
            <div className="upload-divider">
              <span>OR UPLOAD IMAGE</span>
            </div>

            {/* Clickable dashed dropzone */}
            <label className="upload-dropzone">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setImageFile(e.target.files[0] || null)}
              />

              {/* Simple upload icon */}
              <svg
                className="upload-icon"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  d="M12 16V4m0 0L7 9m5-5 5 5M5 20h14"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>

              <span className="upload-title">Choose File</span>
              <span className="upload-subtitle">PNG / JPG recommended</span>
            </label>
          </div>
          <div
            style={{
              marginTop: "1rem",
              display: "flex",
              gap: "0.5rem",
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            <button onClick={handleLoadImageClick} disabled={isRunning}>
              Load Image
            </button>
            <button
              className="secondary"
              onClick={runAnalysis}
              disabled={!canRun || isRunning}
            >
              {isRunning ? (
                <>
                  <span className="spinner" />
                  Running...
                </>
              ) : (
                "Run Transforms & Analysis"
              )}
            </button>
            <button
              className="secondary"
              type="button"
              onClick={async () => {
                if (!imgElementRef.current) {
                  setStatus("Load an image first, then run the sweep.");
                  return;
                }
                setIsSweepRunning(true);
                setStatus("Running sweep experiments...");
                try {
                  const results = await runSweepForCurrentImage({
                    imgEl: imgElementRef.current,
                    hiddenCanvas: hiddenCanvasRef.current,
                    blockSizes: [4, 8, 16, 32], // tweak as you like
                    kFractions: [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
                  });

                  console.table(results);

                  const header = "blockSize,k,rate,transform,psnr\n";
                  const rows = results
                    .map(
                      (r) =>
                        `${r.blockSize},${r.k},${r.rate.toFixed(4)},${
                          r.transform
                        },${r.psnr.toFixed(4)}`
                    )
                    .join("\n");
                  const blob = new Blob([header + rows], { type: "text/csv" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = "transform_sweep_results.csv";
                  a.click();
                  URL.revokeObjectURL(url);

                  setStatus(
                    "Sweep done. CSV downloaded and data logged to console."
                  );
                } catch (err) {
                  console.error(err);
                  setStatus("Error during sweep: " + err.message);
                } finally {
                  setIsSweepRunning(false);
                }
              }}
              disabled={isRunning || isSweepRunning || !canRun}
            >
              {isSweepRunning ? (
                <>
                  <span className="spinner" />
                  Running sweep...
                </>
              ) : (
                "Run Sweep (for current image)"
              )}
            </button>

            <span className="small-text">
              <b>
                Note. Run sweeps button performs a sweep over block sizes (4×4
                up to 64×64) and random k values to collect PSNR for DCT,
                Hadamard, and PCA on this image.
              </b>
            </span>
            <span className="small-text">{status}</span>
          </div>
        </section>

        {/* 2. Original & reconstructions */}
        <section className="card">
          <h2>
            <span className="blue-button">2.</span> Original & Reconstructed
            Images
          </h2>
          <div className="grid-3">
            <div>
              <p className="small-text">Original (grayscale)</p>
              <canvas ref={originalCanvasRef} className="preview" />
            </div>
            <div>
              <p className="small-text">DCT Reconstruction</p>
              <canvas ref={dctCanvasRef} className="preview" />
            </div>
            <div>
              <p className="small-text">Hadamard Reconstruction</p>
              <canvas ref={hadCanvasRef} className="preview" />
            </div>
          </div>
          <div style={{ marginTop: "1rem" }} className="grid-3">
            <div>
              <p className="small-text">
                PCA (Learned Transform) Reconstruction
              </p>
              <canvas ref={pcaCanvasRef} className="preview" />
            </div>
          </div>

          {metrics && (
            <div className="metrics-highlight">
              <div className="metrics-header">
                <span className="metrics-pill">Key output</span>
                <span className="metrics-rate">
                  Rate ≈ k/d = {(metrics.kShow / metrics.d).toFixed(3)}
                </span>
              </div>

              <h3 className="metrics-title">
                Metrics at k = {metrics.kShow}{" "}
                <span className="metrics-subtitle">
                  (kept coeffs/block, d = {metrics.d})
                </span>
              </h3>

              <ul className="metrics-list">
                <li>
                  <span className="metrics-label">DCT PSNR</span>
                  <span className="metrics-value">
                    {metrics.P_dct.toFixed(3)} dB
                  </span>
                </li>
                <li>
                  <span className="metrics-label">Hadamard PSNR</span>
                  <span className="metrics-value">
                    {metrics.P_had.toFixed(3)} dB
                  </span>
                </li>
                <li>
                  <span className="metrics-label">PCA (learned) PSNR</span>
                  <span className="metrics-value">
                    {metrics.P_pca.toFixed(3)} dB
                  </span>
                </li>
              </ul>
            </div>
          )}
        </section>

        {/* 3. Plots */}
        <section className="card">
          <h2>
            <span className="blue-button">3.</span> Energy Compaction &
            Rate–Distortion
          </h2>
          <div className="grid-2">
            <div>
              <p className="small-text">
                Energy compaction: how quickly each transform packs energy into
                top-k coefficients.
              </p>
              <canvas ref={energyCanvasRef} className="preview"/>
            </div>
            <div>
              <p className="small-text">
                Rate–distortion: PSNR vs fraction of kept coefficients per
                block.
              </p>
              <canvas ref={rdCanvasRef} className="preview" />
            </div>
          </div>
        </section>

        {/* 4. Matrices */}
        <section className="card">
          <h2>
            <span className="blue-button">4.</span> Transform Matrices (N×N for
            illustration)
          </h2>
          <div className="grid-3">
            <div>
              <p className="small-text">DCT 1D (N×N)</p>
              <div className="matrix-wrapper">
                <table className="matrix">
                  <tbody>
                    {dctMatrix &&
                      dctMatrix.map((row, i) => (
                        <tr key={i}>
                          {row.map((v, j) => (
                            <td key={j}>{v.toFixed(3)}</td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div>
              <p className="small-text">Hadamard 1D (N×N)</p>
              <div className="matrix-wrapper">
                <table className="matrix">
                  <tbody>
                    {hadMatrix &&
                      hadMatrix.map((row, i) => (
                        <tr key={i}>
                          {row.map((v, j) => (
                            <td key={j}>{v.toFixed(1)}</td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div>
              <p className="small-text">
                PCA: first basis vector reshaped to N×N
              </p>
              <canvas ref={pcaBasisCanvasRef} className="preview" />
            </div>
          </div>
        </section>
              <section className="card">
        <h2>About the Creator</h2>
        <p>
          This project was designed and developed by <strong>Yashika Ahlawat </strong> 
          in partial fulfillment of the requirements for the degree of Master of Science Graduate Program in Computer Science GWU.
        </p>
      </section>
      </main>

      {/* Hidden working canvas */}
      <canvas ref={hiddenCanvasRef} className="preview" style={{ display: "none" }} />

      <footer>
        <div>
          © 2025 Yashika Ahlawat · Transform-Based Image Compression Explorer · George Washington University
        </div>
      </footer>
    </div>
  );
}
