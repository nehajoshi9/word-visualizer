export interface WordItem {
  id: string;
  word: string;
  vector: number[];
  x: number;
  y: number;
  color?: string;
  addedAt: number;
}

export interface SearchResult {
  word: string;
  score: number; // 0 to 1
  isAlreadyOnCanvas: boolean;
  minScore: number;
  avgScore: number;
}

export interface Preset {
  id: string;
  title: string;
  description: string;
  words: string[];
  category: string;
}

export interface ViewportState {
  panX: number;
  panY: number;
  zoom: number;
}

export interface PCAResult {
  points: { id: string; x: number; y: number }[];
  explainedVariance: [number, number]; // variance ratio for PC1 and PC2
}
