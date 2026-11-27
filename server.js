import http from 'http';
import { DatabaseSync } from 'node:sqlite';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DB_PATH = path.join(__dirname, 'words.db');
const PORT = process.env.PORT || 3001;
const LRU_CAPACITY = 40000;
const DIM = 384;

// 1. Initialize SQLite connection
const db = new DatabaseSync(DB_PATH);

// Prepared statements for high performance
const stmtGetWord = db.prepare('SELECT word, vector, freq_rank, access_count FROM dictionary WHERE word = ?');
const stmtUpdateWordVector = db.prepare(`
  INSERT INTO dictionary (word, vector, access_count, last_accessed)
  VALUES (?, ?, 1, ?)
  ON CONFLICT(word) DO UPDATE SET
    vector = excluded.vector,
    access_count = dictionary.access_count + 1,
    last_accessed = excluded.last_accessed
`);
const stmtIncrementAccess = db.prepare(`
  UPDATE dictionary SET access_count = access_count + 1, last_accessed = ? WHERE word = ?
`);
const stmtGetTopFrequent = db.prepare(`
  SELECT word, vector FROM dictionary
  WHERE vector IS NOT NULL
  ORDER BY freq_rank ASC
  LIMIT ?
`);
const stmtGetStats = db.prepare(`
  SELECT COUNT(*) as total_words, COUNT(vector) as embedded_words FROM dictionary
`);

// 2. High-performance O(1) LRU Cache
class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.cache = new Map();
    this.hits = 0;
    this.misses = 0;
  }

  get(key) {
    const val = this.cache.get(key);
    if (!val) {
      this.misses++;
      return undefined;
    }
    this.hits++;
    // Move to most recently used (end of Map)
    this.cache.delete(key);
    this.cache.set(key, val);
    return val;
  }

  set(key, val) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.capacity) {
      // Evict least recently used (first key in Map)
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }
    this.cache.set(key, val);
  }

  has(key) {
    return this.cache.has(key);
  }

  size() {
    return this.cache.size;
  }

  entries() {
    return this.cache.entries();
  }
}

const lruCache = new LRUCache(LRU_CAPACITY);

// 3. Lazy-load Transformers.js embedder
let pipelineExtractor = null;
async function getExtractor() {
  if (!pipelineExtractor) {
    const { pipeline, env } = await import('@xenova/transformers');
    env.allowRemoteModels = false;
    pipelineExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      quantized: true,
    });
  }
  return pipelineExtractor;
}

// 4. Preload LRU cache with the 8,000 most frequently used English words on startup
console.log(`Preloading LRU Cache with top ${LRU_CAPACITY} most frequent English words from SQLite...`);
const rows = stmtGetTopFrequent.all(LRU_CAPACITY);
for (const row of rows) {
  const f32 = new Float32Array(row.vector.buffer, row.vector.byteOffset, row.vector.byteLength / 4);
  lruCache.set(row.word, f32);
}
console.log(`LRU Cache initialized with ${lruCache.size()} words!`);

// 5. Embedding lookup: LRU -> SQLite -> Transformers.js
async function getOrComputeVector(word) {
  const clean = word.toLowerCase().trim();
  if (!clean) throw new Error('Word is empty');

  // Step 1: Check LRU Cache
  const cached = lruCache.get(clean);
  if (cached) {
    return { vector: Array.from(cached), source: 'lru' };
  }

  const now = Date.now();

  // Step 2: Check SQLite Database
  const dbRow = stmtGetWord.get(clean);
  if (dbRow && dbRow.vector) {
    const f32 = new Float32Array(dbRow.vector.buffer, dbRow.vector.byteOffset, dbRow.vector.byteLength / 4);
    lruCache.set(clean, f32);
    stmtIncrementAccess.run(now, clean);
    return { vector: Array.from(f32), source: 'sqlite' };
  }

  // Step 3: Compute on the fly with neural model
  const extractor = await getExtractor();
  const output = await extractor(clean, { pooling: 'mean', normalize: true });
  const f32 = new Float32Array(output.data);

  // Persist to SQLite
  stmtUpdateWordVector.run(clean, Buffer.from(f32.buffer), now);

  // Put into LRU cache
  lruCache.set(clean, f32);

  return { vector: Array.from(f32), source: 'computed' };
}

// 6. Dot product and similarity helper
function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < DIM; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function findSimilarWords(selectedVectors, selectedWords, limit = 30) {
  const selSet = new Set(selectedWords.map(s => s.toLowerCase()));
  const numSel = selectedVectors.length;
  const scored = [];

  // Iterate over active LRU cache entries
  for (const [w, vec] of lruCache.entries()) {
    if (selSet.has(w)) continue;

    let minSim = 1.0;
    let sumSim = 0;

    for (let s = 0; s < numSel; s++) {
      const d = dot(selectedVectors[s], vec);
      if (d < minSim) minSim = d;
      sumSim += d;
    }

    const avgSim = sumSim / numSel;
    const composite = numSel === 1 ? avgSim : 0.55 * avgSim + 0.45 * minSim;
    const normalizedScore = Math.max(0, Math.min(1, (composite + 0.05) / 0.85));

    scored.push({
      word: w,
      score: normalizedScore,
      minScore: Math.max(0, Math.min(1, (minSim + 0.05) / 0.85)),
      avgScore: Math.max(0, Math.min(1, (avgSim + 0.05) / 0.85)),
    });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

// 7. HTTP API Server
const server = http.createServer(async (req, res) => {
  // CORS Headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url || '/', `http://${req.headers.host}`);

  // Helper to read JSON body
  const readJsonBody = async () => {
    return new Promise((resolve, reject) => {
      let body = '';
      req.on('data', chunk => (body += chunk));
      req.on('end', () => {
        try {
          resolve(body ? JSON.parse(body) : {});
        } catch (err) {
          reject(err);
        }
      });
      req.on('error', reject);
    });
  };

  try {
    if (req.method === 'POST' && url.pathname === '/api/embed') {
      const body = await readJsonBody();
      const word = body.word;
      if (!word || typeof word !== 'string') {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing word parameter' }));
        return;
      }

      const result = await getOrComputeVector(word);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ word: word.toLowerCase().trim(), ...result }));
      return;
    }

    if (req.method === 'POST' && url.pathname === '/api/similar') {
      const body = await readJsonBody();
      const { selectedWords, limit = 30 } = body;
      if (!Array.isArray(selectedWords) || selectedWords.length === 0) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ results: [] }));
        return;
      }

      // Fetch or compute vectors for all selected words
      const selVectors = [];
      const validSelectedWords = [];
      for (const w of selectedWords) {
        try {
          const resVec = await getOrComputeVector(w);
          selVectors.push(resVec.vector);
          validSelectedWords.push(w);
        } catch (e) {
          console.warn('Could not embed selected word:', w, e);
        }
      }

      const results = findSimilarWords(selVectors, validSelectedWords, limit);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ results }));
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/stats') {
      const stats = db.prepare('SELECT COUNT(*) as total_words, COUNT(vector) as embedded_words FROM dictionary').get();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          lruCacheSize: lruCache.size(),
          lruCapacity: LRU_CAPACITY,
          lruHits: lruCache.hits,
          lruMisses: lruCache.misses,
          dbTotalWords: stats.total_words,
          dbEmbeddedWords: stats.embedded_words,
        })
      );
      return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  } catch (err) {
    console.error('Server error:', err);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: err.message || 'Internal server error' }));
  }
});

server.listen(PORT, () => {
  console.log(`Backend SQLite + LRU Server listening on http://localhost:${PORT}`);
});
