import { SearchResult, WordItem } from '../types';

/**
 * Computes dot product of two vectors
 */
export function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Computes magnitude (L2 norm) of a vector
 */
export function norm(v: number[]): number {
  let sumSq = 0;
  for (let i = 0; i < v.length; i++) {
    sumSq += v[i] * v[i];
  }
  return Math.sqrt(sumSq);
}

/**
 * Computes cosine similarity between two vectors (-1 to 1)
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  const normA = norm(a);
  const normB = norm(b);
  if (normA === 0 || normB === 0) return 0;
  return dotProduct(a, b) / (normA * normB);
}

/**
 * Normalizes vector to unit length
 */
export function normalizeVector(v: number[]): number[] {
  const n = norm(v);
  if (n === 0) return [...v];
  return v.map(x => x / n);
}

/**
 * Computes centroid vector of multiple vectors
 */
export function computeCentroid(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dim = vectors[0].length;
  const result = new Array(dim).fill(0);
  for (const v of vectors) {
    const nv = normalizeVector(v);
    for (let d = 0; d < dim; d++) {
      result[d] += nv[d];
    }
  }
  for (let d = 0; d < dim; d++) {
    result[d] /= vectors.length;
  }
  return normalizeVector(result);
}

/**
 * Finds words in the vocabulary that have the highest semantic similarity
 * to ALL selected words simultaneously.
 */
export function findSimilarToAll(
  selectedWords: WordItem[],
  vocabulary: Record<string, number[]>,
  canvasWords: Set<string>,
  limit = 25
): SearchResult[] {
  if (selectedWords.length === 0) return [];

  const selectedNormVectors = selectedWords.map(w => normalizeVector(w.vector));
  const selectedWordSet = new Set(selectedWords.map(w => w.word.toLowerCase()));

  const results: SearchResult[] = [];

  for (const [word, vector] of Object.entries(vocabulary)) {
    const wordLower = word.toLowerCase();
    // Don't recommend the selected words themselves
    if (selectedWordSet.has(wordLower)) continue;

    const vNorm = normalizeVector(vector);

    let minSim = 1.0;
    let sumSim = 0;

    for (const sVec of selectedNormVectors) {
      const sim = dotProduct(sVec, vNorm);
      if (sim < minSim) minSim = sim;
      sumSim += sim;
    }

    const avgSim = sumSim / selectedNormVectors.length;

    // Combined score: heavily rewards candidates that stay close to ALL selected words
    // If multiple words are selected, penalty is applied if candidate is far from any single word
    let compositeScore: number;
    if (selectedNormVectors.length === 1) {
      compositeScore = avgSim;
    } else {
      // 60% average similarity + 40% minimum similarity to prevent outliers
      compositeScore = 0.6 * avgSim + 0.4 * minSim;
    }

    // Convert -1..1 cosine to a normalized 0..1 scale for user display
    // Usually semantic cosine for Glove/embeddings is in range [0, 1] for related words
    const displayScore = Math.max(0, Math.min(1, (compositeScore + 0.2) / 1.2));

    results.push({
      word,
      score: displayScore,
      minScore: Math.max(0, Math.min(1, (minSim + 0.2) / 1.2)),
      avgScore: Math.max(0, Math.min(1, (avgSim + 0.2) / 1.2)),
      isAlreadyOnCanvas: canvasWords.has(wordLower),
    });
  }

  // Sort descending by composite score
  results.sort((a, b) => b.score - a.score);

  return results.slice(0, limit);
}
