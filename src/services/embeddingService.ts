// Hybrid Embedding Service:
// Automatically uses the local SQLite/LRU backend if available (/api/embed),
// OR seamlessly runs 100% in the browser (static GitHub Pages / Vercel / Cloudflare Pages)
// with zero servers and zero database requirements (no Supabase needed).

export interface SystemStats {
  lruCacheSize: number;
  lruCapacity: number;
  dbTotalWords: number;
  dbEmbeddedWords: number;
  mode: 'server' | 'browser';
}

export interface SimilarityResult {
  word: string;
  score: number;
  minScore: number;
  avgScore: number;
  isAlreadyOnCanvas: boolean;
}

const LRU_CAPACITY = 8000;
const DIM = 384;

class LRUCache {
  private capacity: number;
  private cache: Map<string, Float32Array>;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.cache = new Map();
  }

  get(key: string): Float32Array | undefined {
    const val = this.cache.get(key);
    if (!val) return undefined;
    this.cache.delete(key);
    this.cache.set(key, val);
    return val;
  }

  set(key: string, val: Float32Array): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.capacity) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) this.cache.delete(oldestKey);
    }
    this.cache.set(key, val);
  }

  size(): number {
    return this.cache.size;
  }

  entries(): IterableIterator<[string, Float32Array]> {
    return this.cache.entries();
  }
}

class EmbeddingService {
  private isServerAvailable: boolean | null = null;
  private clientLRUCache = new LRUCache(LRU_CAPACITY);
  private vocabLoaded = false;
  private vocabWords: string[] = [];
  private vocabWordToIndex: Map<string, number> = new Map();
  private vocabVectors: Float32Array | null = null;
  private pipelinePromise: Promise<any> | null = null;

  async init(): Promise<void> {
    if (this.isServerAvailable !== null) return;

    try {
      // Test if local server is running
      const resp = await fetch('/api/stats', { signal: AbortSignal.timeout(1200) });
      if (resp.ok) {
        this.isServerAvailable = true;
        console.log('[EmbeddingService] Connected to backend server.');
        return;
      }
    } catch {
      // Server not reachable -> running on static hosting (GitHub Pages, Vercel, etc.)
    }

    this.isServerAvailable = false;
    console.log('[EmbeddingService] Running in standalone browser mode (no server needed).');
    await this.initBrowserMode();
  }

  private async initBrowserMode(): Promise<void> {
    if (this.vocabLoaded) return;
    try {
      const baseUrl = import.meta.env.BASE_URL || '/';
      const cleanBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;

      const [wordsResp, binResp] = await Promise.all([
        fetch(`${cleanBase}vocab_words.json`),
        fetch(`${cleanBase}vocab_vectors.bin`),
      ]);

      if (!wordsResp.ok || !binResp.ok) {
        throw new Error('Static vocabulary assets not found');
      }

      this.vocabWords = await wordsResp.json();
      this.vocabWords.forEach((w, i) => this.vocabWordToIndex.set(w.toLowerCase(), i));

      const arrayBuf = await binResp.arrayBuffer();
      this.vocabVectors = new Float32Array(arrayBuf);

      // Preload the 8,000 most frequent words into client LRU Cache
      const count = Math.min(this.vocabWords.length, LRU_CAPACITY);
      for (let i = 0; i < count; i++) {
        const w = this.vocabWords[i];
        const sub = this.vocabVectors.subarray(i * DIM, (i + 1) * DIM);
        this.clientLRUCache.set(w, sub);
      }

      this.vocabLoaded = true;
      console.log(`[EmbeddingService] Loaded ${count} words into in-memory browser LRU cache.`);
    } catch (err) {
      console.warn('[EmbeddingService] Standalone browser init warning:', err);
    }
  }

  // Lazy-load Transformers.js in WebAssembly if user types a novel word
  private async getBrowserExtractor(): Promise<any> {
    if (!this.pipelinePromise) {
      this.pipelinePromise = (async () => {
        const { pipeline, env } = await import('@xenova/transformers');
        env.allowLocalModels = false;
        env.useBrowserCache = true;
        return await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
          quantized: true,
        });
      })();
    }
    return this.pipelinePromise;
  }

  async getStats(): Promise<SystemStats | null> {
    await this.init();

    if (this.isServerAvailable) {
      try {
        const resp = await fetch('/api/stats');
        if (resp.ok) {
          const data = await resp.json();
          return { ...data, mode: 'server' };
        }
      } catch {
        this.isServerAvailable = false;
      }
    }

    return {
      lruCacheSize: this.clientLRUCache.size(),
      lruCapacity: LRU_CAPACITY,
      dbTotalWords: this.vocabWords.length || 8000,
      dbEmbeddedWords: this.clientLRUCache.size(),
      mode: 'browser',
    };
  }

  async getWordVector(word: string): Promise<number[]> {
    await this.init();
    const clean = word.toLowerCase().trim();
    if (!clean) throw new Error('Word cannot be empty');

    // 1. If server available, use server
    if (this.isServerAvailable) {
      try {
        const resp = await fetch('/api/embed', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ word: clean }),
        });

        if (resp.ok) {
          const data = await resp.json();
          return data.vector;
        }
      } catch {
        this.isServerAvailable = false;
      }
    }

    // 2. Standalone browser mode
    // Check client LRU cache
    const cached = this.clientLRUCache.get(clean);
    if (cached) return Array.from(cached);

    // Check precomputed vocabulary
    const idx = this.vocabWordToIndex.get(clean);
    if (idx !== undefined && this.vocabVectors) {
      const vec = this.vocabVectors.subarray(idx * DIM, (idx + 1) * DIM);
      this.clientLRUCache.set(clean, vec);
      return Array.from(vec);
    }

    // Embed on the fly in the browser via WebAssembly
    const extractor = await this.getBrowserExtractor();
    const output = await extractor(clean, { pooling: 'mean', normalize: true });
    const f32 = new Float32Array(output.data);
    this.clientLRUCache.set(clean, f32);
    return Array.from(f32);
  }

  async findSimilarToAllWords(
    selectedWords: { word: string; vector: number[] }[],
    canvasWordsSet: Set<string>,
    limit = 20
  ): Promise<SimilarityResult[]> {
    if (selectedWords.length === 0) return [];
    await this.init();

    // 1. Try server endpoint if active
    if (this.isServerAvailable) {
      try {
        const resp = await fetch('/api/similar', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            selectedWords: selectedWords.map(s => s.word),
            limit,
          }),
        });

        if (resp.ok) {
          const data = await resp.json();
          return (data.results || []).map((r: any) => ({
            ...r,
            isAlreadyOnCanvas: canvasWordsSet.has(r.word.toLowerCase()),
          }));
        }
      } catch {
        this.isServerAvailable = false;
      }
    }

    // 2. Standalone in-browser similarity search
    const selSet = new Set(selectedWords.map(s => s.word.toLowerCase()));
    const selVecs = selectedWords.map(s => s.vector);
    const numSel = selVecs.length;
    const scored: SimilarityResult[] = [];

    for (const [w, vec] of this.clientLRUCache.entries()) {
      if (selSet.has(w)) continue;

      let minSim = 1.0;
      let sumSim = 0;

      for (let s = 0; s < numSel; s++) {
        const sVec = selVecs[s];
        let dot = 0;
        for (let d = 0; d < DIM; d++) dot += sVec[d] * vec[d];
        if (dot < minSim) minSim = dot;
        sumSim += dot;
      }

      const avgSim = sumSim / numSel;
      const composite = numSel === 1 ? avgSim : 0.55 * avgSim + 0.45 * minSim;
      const normalizedScore = Math.max(0, Math.min(1, (composite + 0.05) / 0.85));

      scored.push({
        word: w,
        score: normalizedScore,
        minScore: Math.max(0, Math.min(1, (minSim + 0.05) / 0.85)),
        avgScore: Math.max(0, Math.min(1, (avgSim + 0.05) / 0.85)),
        isAlreadyOnCanvas: canvasWordsSet.has(w),
      });
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }
}

export const embeddingService = new EmbeddingService();
