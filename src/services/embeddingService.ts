// Fully automatic, zero-UI 3-layer sharded vocabulary architecture:
// Layer 1: Immediate Core 8k Boot (Int8 quantized, ~3 MB)
// Layer 2: Silent Background Idle Upgrade (32k Extended, Int8 quantized, ~12 MB) with IndexedDB persistence
// Layer 3: Dynamic In-Browser WASM Fallback (@xenova/transformers) for rare/novel words

export interface SystemStats {
  totalWords: number;
  lruCacheSize: number;
  lruCapacity: number;
  dbTotalWords: number;
  dbEmbeddedWords: number;
  mode: 'server' | 'browser';
  loadedShards?: string[];
}

export interface SimilarityResult {
  word: string;
  score: number;
  minScore: number;
  avgScore: number;
  isAlreadyOnCanvas: boolean;
}

export interface LoadedShard {
  id: string;
  words: string[];
  vectors: Int8Array;
  dim: number;
}

const DIM = 384;
const IDB_NAME = 'word-visualizer-shards';
const IDB_STORE = 'shards';
const IDB_VERSION = 1;

function getBaseUrl(): string {
  const url = (typeof import.meta !== 'undefined' && (import.meta as any).env?.BASE_URL) || './';
  return url.endsWith('/') ? url : `${url}/`;
}

// IndexedDB persistence helpers
function openIDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      return reject(new Error('IndexedDB not supported in this environment'));
    }
    const req = indexedDB.open(IDB_NAME, IDB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function getShardFromIDB(id: string): Promise<{ words: string[]; buffer: ArrayBuffer } | null> {
  try {
    const db = await openIDB();
    return new Promise((resolve) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const store = tx.objectStore(IDB_STORE);
      const req = store.get(id);
      req.onsuccess = () => {
        if (req.result && req.result.words && req.result.buffer) {
          resolve({ words: req.result.words, buffer: req.result.buffer });
        } else {
          resolve(null);
        }
      };
      req.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

async function saveShardToIDB(id: string, words: string[], buffer: ArrayBuffer): Promise<void> {
  try {
    const db = await openIDB();
    return new Promise((resolve) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      const store = tx.objectStore(IDB_STORE);
      store.put({ id, words, buffer });
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch {
    // Ignore IDB write failures (e.g. storage quota / incognito restrictions)
  }
}

class EmbeddingService {
  private isServerAvailable: boolean | null = null;
  private shards: LoadedShard[] = [];
  // Unified O(1) word lookup map: word -> { shardIdx, rowIdx }
  private unifiedIndex: Map<string, { shardIdx: number; rowIdx: number }> = new Map();
  // Layer 3 session fallback pool for on-the-fly WASM-embedded words
  private fallbackWords: Map<string, { f32: Float32Array; int8: Int8Array }> = new Map();

  private isInitPromise: Promise<void> | null = null;
  private isExtendedShardLoaded = false;
  private extendedShardLoading = false;
  private pipelinePromise: Promise<any> | null = null;

  async init(): Promise<void> {
    if (this.isInitPromise) return this.isInitPromise;
    this.isInitPromise = this.boot();
    return this.isInitPromise;
  }

  private async boot(): Promise<void> {
    if (this.isServerAvailable !== null) return;

    try {
      // Test if local server is running (e.g. during local dev)
      const resp = await fetch('/api/stats', { signal: AbortSignal.timeout(800) });
      if (resp.ok) {
        this.isServerAvailable = true;
        console.log('[EmbeddingService] Connected to backend server.');
        return;
      }
    } catch {
      // Server not reachable -> running on static hosting (GitHub Pages, Vercel, etc.)
    }

    this.isServerAvailable = false;
    console.log('[EmbeddingService] Running in standalone 3-layer sharded browser mode.');
    await this.bootCoreShard();
  }

  // Layer 1 (Immediate Boot): Load Core 8k shard
  private async bootCoreShard(): Promise<void> {
    if (this.shards.some(s => s.id === 'core-8k-v1')) return;

    try {
      const cleanBase = getBaseUrl();

      const [wordsResp, binResp] = await Promise.all([
        fetch(`${cleanBase}vectors-core-8k.v1.json`),
        fetch(`${cleanBase}vectors-core-8k.v1.bin`),
      ]);

      if (!wordsResp.ok || !binResp.ok) {
        throw new Error(`Failed to load Core 8k shard: words=${wordsResp.status}, bin=${binResp.status}`);
      }

      const coreWords: string[] = await wordsResp.json();
      const arrayBuf = await binResp.arrayBuffer();
      const coreInt8 = new Int8Array(arrayBuf);

      const shardIdx = this.shards.length;
      this.shards.push({
        id: 'core-8k-v1',
        words: coreWords,
        vectors: coreInt8,
        dim: DIM,
      });

      for (let i = 0; i < coreWords.length; i++) {
        this.unifiedIndex.set(coreWords[i].toLowerCase(), { shardIdx, rowIdx: i });
      }

      console.log(`[EmbeddingService] Layer 1 Ready: Core 8k shard active (${coreWords.length} words).`);

      // Trigger Layer 2: Silent background idle upgrade
      this.scheduleIdleExtendedShard();
    } catch (err) {
      console.warn('[EmbeddingService] Core 8k shard boot warning:', err);
    }
  }

  // Layer 2 (Silent Idle Upgrade + IndexedDB Persistence)
  private scheduleIdleExtendedShard(): void {
    if (this.extendedShardLoading || this.isExtendedShardLoaded) return;

    const loadFn = () => {
      this.loadExtendedShard().catch(err => {
        console.warn('[EmbeddingService] Background extended shard upgrade warning:', err);
      });
    };

    if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
      (window as any).requestIdleCallback(loadFn, { timeout: 3000 });
    } else {
      setTimeout(loadFn, 1500);
    }
  }

  private async loadExtendedShard(): Promise<void> {
    const SHARD_ID = 'ext-32k-v1';
    if (this.extendedShardLoading || this.isExtendedShardLoaded || this.shards.some(s => s.id === SHARD_ID)) {
      return;
    }
    this.extendedShardLoading = true;

    try {
      let extWords: string[] | null = null;
      let extBuffer: ArrayBuffer | null = null;

      // 1. Check IndexedDB cache first (~40ms, zero network)
      try {
        const cached = await getShardFromIDB(SHARD_ID);
        if (cached && cached.words && cached.buffer) {
          extWords = cached.words;
          extBuffer = cached.buffer;
          console.log(`[EmbeddingService] Layer 2: Loaded Extended 32k shard from IndexedDB cache (~40ms, zero network).`);
        }
      } catch (err) {
        console.warn('[EmbeddingService] IndexedDB read warning:', err);
      }

      // 2. If not cached, fetch in background and persist
      if (!extWords || !extBuffer) {
        const cleanBase = getBaseUrl();

        const [wordsResp, binResp] = await Promise.all([
          fetch(`${cleanBase}vectors-ext-32k.v1.json`),
          fetch(`${cleanBase}vectors-ext-32k.v1.bin`),
        ]);

        if (!wordsResp.ok || !binResp.ok) {
          console.warn('[EmbeddingService] Extended shard assets not found on server.');
          return;
        }

        extWords = await wordsResp.json();
        extBuffer = await binResp.arrayBuffer();

        // Save to IndexedDB for instant load on subsequent visits
        saveShardToIDB(SHARD_ID, extWords!, extBuffer!).catch(() => {});
        console.log(`[EmbeddingService] Layer 2: Downloaded Extended 32k shard & cached in IndexedDB.`);
      }

      // 3. Append to active shard pool
      if (!extWords || !extBuffer) return;

      const extInt8 = new Int8Array(extBuffer);
      const shardIdx = this.shards.length;
      this.shards.push({
        id: SHARD_ID,
        words: extWords,
        vectors: extInt8,
        dim: DIM,
      });

      // Update unified index and cleanly deduplicate any session WASM fallback words
      for (let i = 0; i < extWords.length; i++) {
        const w = extWords[i].toLowerCase();
        if (!this.unifiedIndex.has(w)) {
          this.unifiedIndex.set(w, { shardIdx, rowIdx: i });
        }
        if (this.fallbackWords.has(w)) {
          this.fallbackWords.delete(w);
        }
      }

      this.isExtendedShardLoaded = true;
      console.log(`[EmbeddingService] Layer 2 Active: Shard pool upgraded to ${this.unifiedIndex.size} total words.`);
    } finally {
      this.extendedShardLoading = false;
    }
  }

  // Dequantizes Int8 row (val / 127) when exact Float32 vectors are needed for PCA or distances
  getNormalizedFloat32(word: string): Float32Array | null {
    const clean = word.toLowerCase().trim();
    if (!clean) return null;

    const loc = this.unifiedIndex.get(clean);
    if (loc) {
      const shard = this.shards[loc.shardIdx];
      if (!shard) return null;
      const offset = loc.rowIdx * shard.dim;
      const f32 = new Float32Array(shard.dim);
      for (let d = 0; d < shard.dim; d++) {
        f32[d] = shard.vectors[offset + d] / 127;
      }
      return f32;
    }

    const fallback = this.fallbackWords.get(clean);
    if (fallback) {
      return fallback.f32;
    }

    return null;
  }

  // Lazy-load Transformers.js in WebAssembly if user types a novel word
  private async getBrowserExtractor(): Promise<any> {
    if (!this.pipelinePromise) {
      this.pipelinePromise = (async () => {
        const { pipeline, env } = await import('@xenova/transformers');
        env.allowLocalModels = typeof process !== 'undefined' && !!process.versions?.node;
        env.useBrowserCache = typeof caches !== 'undefined';
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

    const totalWords = this.unifiedIndex.size + this.fallbackWords.size;
    return {
      totalWords,
      lruCacheSize: totalWords,
      lruCapacity: 40000,
      dbTotalWords: totalWords,
      dbEmbeddedWords: totalWords,
      mode: 'browser',
      loadedShards: this.shards.map(s => s.id),
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

    // 2. Standalone browser mode: Check loaded shards & session fallback pool
    const existingF32 = this.getNormalizedFloat32(clean);
    if (existingF32) {
      return Array.from(existingF32);
    }

    // 3. Layer 3: On-the-Fly WASM Fallback
    console.log(`[EmbeddingService] Layer 3 WASM Fallback: Embedding "${clean}" on-the-fly...`);
    const extractor = await this.getBrowserExtractor();
    const output = await extractor(clean, { pooling: 'mean', normalize: true });
    const f32 = new Float32Array(output.data);

    // Quantize to Int8 for affinity scans
    const int8 = new Int8Array(DIM);
    for (let d = 0; d < DIM; d++) {
      int8[d] = Math.round(Math.max(-1, Math.min(1, f32[d])) * 127);
    }

    // Store in session fallback pool
    this.fallbackWords.set(clean, { f32, int8 });
    return Array.from(f32);
  }

  // Fast Multi-Shard Cosine Similarity Scan over Int8Array buffers
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

    // 2. Fast multi-shard cosine similarity scan in browser
    const numSel = selectedWords.length;
    const selSet = new Set(selectedWords.map(s => s.word.toLowerCase()));

    // Pre-calculate scaled query vectors (scaled by 1 / 127) for direct Int8 dot products
    const scaledQueries: Float32Array[] = [];
    const rawQueries: Float32Array[] = [];

    for (let s = 0; s < numSel; s++) {
      const raw = new Float32Array(selectedWords[s].vector);
      rawQueries.push(raw);
      const scaled = new Float32Array(DIM);
      for (let d = 0; d < DIM; d++) {
        scaled[d] = raw[d] / 127;
      }
      scaledQueries.push(scaled);
    }

    const scored: SimilarityResult[] = [];

    // Iterate through all loaded shards (Core 8k + Extended 32k if ready)
    for (let sIdx = 0; sIdx < this.shards.length; sIdx++) {
      const shard = this.shards[sIdx];
      const words = shard.words;
      const vectors = shard.vectors;
      const count = words.length;

      for (let i = 0; i < count; i++) {
        const w = words[i].toLowerCase();
        if (selSet.has(w)) continue;

        const offset = i * DIM;
        let minSim = 1.0;
        let sumSim = 0;

        for (let s = 0; s < numSel; s++) {
          const scaledQ = scaledQueries[s];
          let dot = 0;
          for (let d = 0; d < DIM; d++) {
            dot += scaledQ[d] * vectors[offset + d];
          }
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
    }

    // Iterate through session fallback words (WASM embedded novel words)
    for (const [w, { f32 }] of this.fallbackWords.entries()) {
      if (selSet.has(w) || this.unifiedIndex.has(w)) continue;

      let minSim = 1.0;
      let sumSim = 0;

      for (let s = 0; s < numSel; s++) {
        const rawQ = rawQueries[s];
        let dot = 0;
        for (let d = 0; d < DIM; d++) {
          dot += rawQ[d] * f32[d];
        }
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

    // Sort descending by composite affinity score
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }
}

export const embeddingService = new EmbeddingService();
