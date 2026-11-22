import { PCAResult } from '../types';

/**
 * Robust, deterministic Principal Component Analysis (PCA)
 * using the Dual Gram Matrix method (O(N^2)) with deflation.
 * Handles N = 1, 2, and N >= 3 seamlessly.
 */
export function computePCA(
  items: { id: string; vector: number[] }[]
): PCAResult {
  const n = items.length;
  if (n === 0) {
    return { points: [], explainedVariance: [0, 0] };
  }

  if (n === 1) {
    return {
      points: [{ id: items[0].id, x: 0, y: 0 }],
      explainedVariance: [1, 0],
    };
  }

  const dim = items[0].vector.length;

  if (n === 2) {
    // Scale to standard canvas spread
    const spread = 240;
    return {
      points: [
        { id: items[0].id, x: -spread / 2, y: 0 },
        { id: items[1].id, x: spread / 2, y: 0 },
      ],
      explainedVariance: [1, 0],
    };
  }

  // 1. Compute mean vector
  const mean = new Array(dim).fill(0);
  for (let i = 0; i < n; i++) {
    const vec = items[i].vector;
    for (let d = 0; d < dim; d++) {
      mean[d] += vec[d];
    }
  }
  for (let d = 0; d < dim; d++) {
    mean[d] /= n;
  }

  // 2. Center the vectors (Xc)
  const centered: number[][] = [];
  for (let i = 0; i < n; i++) {
    const vec = items[i].vector;
    const row = new Array(dim);
    for (let d = 0; d < dim; d++) {
      row[d] = vec[d] - mean[d];
    }
    centered.push(row);
  }

  // 3. Compute Gram matrix K = Xc * Xc^T (size N x N)
  // This is O(N^2 * D), which is blazingly fast for word lists
  const K: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let dot = 0;
      const rowI = centered[i];
      const rowJ = centered[j];
      for (let d = 0; d < dim; d++) {
        dot += rowI[d] * rowJ[d];
      }
      K[i][j] = dot;
      K[j][i] = dot;
    }
  }

  // 4. Power iteration with deflation to find top 2 eigenvectors and eigenvalues of K
  // Since K u = lambda u, the projection onto PC is z = sqrt(lambda) * u
  const { eigenvalue: lambda1, eigenvector: u1 } = powerIteration(K, n);
  const K_deflated = deflateMatrix(K, u1, lambda1, n);
  const { eigenvalue: lambda2, eigenvector: u2 } = powerIteration(K_deflated, n);

  // Deterministic sign convention (highest absolute component has positive sign)
  fixVectorSign(u1);
  fixVectorSign(u2);

  const scale1 = Math.sqrt(Math.max(0, lambda1));
  const scale2 = Math.sqrt(Math.max(0, lambda2));

  // Compute raw 2D coordinates: z_i = (u1[i] * sqrt(lambda1), u2[i] * sqrt(lambda2))
  const rawPoints = items.map((item, i) => ({
    id: item.id,
    x: u1[i] * scale1,
    y: u2[i] * scale2,
  }));

  // 5. Normalization & canvas fitting
  let maxX = 0;
  let maxY = 0;
  for (const pt of rawPoints) {
    if (Math.abs(pt.x) > maxX) maxX = Math.abs(pt.x);
    if (Math.abs(pt.y) > maxY) maxY = Math.abs(pt.y);
  }

  const maxCoord = Math.max(maxX, maxY, 0.0001);
  const TARGET_RADIUS = 280; // Ideal coordinate range [-280, 280]
  const scaleFactor = TARGET_RADIUS / maxCoord;

  const points = rawPoints.map(pt => ({
    id: pt.id,
    x: Math.round(pt.x * scaleFactor * 10) / 10,
    y: Math.round(pt.y * scaleFactor * 10) / 10,
  }));

  const totalVar = Math.max(0.0001, lambda1 + lambda2);
  const explainedVariance: [number, number] = [
    Math.round((lambda1 / totalVar) * 100) / 100,
    Math.round((lambda2 / totalVar) * 100) / 100,
  ];

  return { points, explainedVariance };
}

function powerIteration(
  A: number[][],
  n: number,
  maxIterations = 80,
  tolerance = 1e-7
): { eigenvalue: number; eigenvector: number[] } {
  // Initial vector with alternating small values to avoid symmetry lock
  let b = new Array(n);
  for (let i = 0; i < n; i++) {
    b[i] = (i % 2 === 0 ? 1 : -1) * (0.5 + (i * 0.1) % 1);
  }
  normalize(b);

  let eigenvalue = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    // Multiply b_next = A * b
    const b_next = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      const row = A[i];
      for (let j = 0; j < n; j++) {
        sum += row[j] * b[j];
      }
      b_next[i] = sum;
    }

    // Rayleigh quotient
    let normSq = 0;
    let bDotBNext = 0;
    for (let i = 0; i < n; i++) {
      normSq += b[i] * b[i];
      bDotBNext += b[i] * b_next[i];
    }
    const currentEigenvalue = normSq > 0 ? bDotBNext / normSq : 0;

    normalize(b_next);

    // Check convergence
    let diff = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(b_next[i] - b[i]);
      if (d > diff) diff = d;
    }

    b = b_next;
    eigenvalue = currentEigenvalue;

    if (diff < tolerance) break;
  }

  return { eigenvalue: Math.max(0, eigenvalue), eigenvector: b };
}

function deflateMatrix(
  A: number[][],
  u: number[],
  lambda: number,
  n: number
): number[][] {
  const result: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      result[i][j] = A[i][j] - lambda * u[i] * u[j];
    }
  }
  return result;
}

function normalize(v: number[]) {
  let sumSq = 0;
  for (let i = 0; i < v.length; i++) {
    sumSq += v[i] * v[i];
  }
  const norm = Math.sqrt(sumSq);
  if (norm > 1e-12) {
    for (let i = 0; i < v.length; i++) {
      v[i] /= norm;
    }
  }
}

function fixVectorSign(v: number[]) {
  // Deterministic orientation: find element with largest absolute value and ensure it's positive
  let maxAbs = 0;
  let sign = 1;
  for (let i = 0; i < v.length; i++) {
    const abs = Math.abs(v[i]);
    if (abs > maxAbs) {
      maxAbs = abs;
      sign = v[i] >= 0 ? 1 : -1;
    }
  }
  if (sign < 0) {
    for (let i = 0; i < v.length; i++) {
      v[i] = -v[i];
    }
  }
}
