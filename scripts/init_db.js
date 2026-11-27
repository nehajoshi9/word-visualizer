import { DatabaseSync } from 'node:sqlite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DB_PATH = path.join(__dirname, '../words.db');
const SYSTEM_DICT = '/usr/share/dict/words';
const CORE_WORDS_PATH = path.join(__dirname, '../public/vectors-core-8k.v1.json');
const CORE_VECTORS_PATH = path.join(__dirname, '../public/vectors-core-8k.v1.bin');
const EXT_WORDS_PATH = path.join(__dirname, '../public/vectors-ext-32k.v1.json');
const EXT_VECTORS_PATH = path.join(__dirname, '../public/vectors-ext-32k.v1.bin');

console.log('Initializing words.db with full English dictionary and 40,000 sharded vocabulary...');

// Remove existing DB if any to start fresh
if (fs.existsSync(DB_PATH)) {
  fs.unlinkSync(DB_PATH);
}

const db = new DatabaseSync(DB_PATH);

// 1. Create schema
db.exec(`
  CREATE TABLE IF NOT EXISTS dictionary (
    word TEXT PRIMARY KEY,
    vector BLOB,
    freq_rank INTEGER,
    access_count INTEGER DEFAULT 0,
    last_accessed INTEGER
  );
  CREATE INDEX IF NOT EXISTS idx_freq_rank ON dictionary (freq_rank);
  CREATE INDEX IF NOT EXISTS idx_has_vector ON dictionary (vector);
`);

const dim = 384;
const insertWordWithVector = db.prepare(`
  INSERT INTO dictionary (word, vector, freq_rank, access_count, last_accessed)
  VALUES (?, ?, ?, 1, ?)
  ON CONFLICT(word) DO UPDATE SET
    vector = excluded.vector,
    freq_rank = excluded.freq_rank,
    last_accessed = excluded.last_accessed
`);

const now = Date.now();

function loadShard(wordsPath, vectorsPath, startRank) {
  if (!fs.existsSync(wordsPath) || !fs.existsSync(vectorsPath)) {
    console.warn(`Shard files not found: ${wordsPath}`);
    return 0;
  }
  const words = JSON.parse(fs.readFileSync(wordsPath, 'utf8'));
  const binBuffer = fs.readFileSync(vectorsPath);
  const int8 = new Int8Array(binBuffer.buffer, binBuffer.byteOffset, binBuffer.byteLength);

  db.exec('BEGIN TRANSACTION');
  const f32Buffer = new Float32Array(dim);
  for (let i = 0; i < words.length; i++) {
    const w = words[i].toLowerCase();
    const offset = i * dim;
    for (let d = 0; d < dim; d++) {
      f32Buffer[d] = int8[offset + d] / 127;
    }
    insertWordWithVector.run(w, Buffer.from(f32Buffer.buffer), startRank + i, now);
  }
  db.exec('COMMIT');
  return words.length;
}

// 2. Load Core 8k and Extended 32k shards if present
let totalEmbedded = 0;
totalEmbedded += loadShard(CORE_WORDS_PATH, CORE_VECTORS_PATH, 1);
totalEmbedded += loadShard(EXT_WORDS_PATH, EXT_VECTORS_PATH, totalEmbedded + 1);
console.log(`Loaded ${totalEmbedded} sharded words with vectors into SQLite.`);

// 3. Load remaining English words from system dictionary
if (fs.existsSync(SYSTEM_DICT)) {
  console.log(`Loading remaining English words from ${SYSTEM_DICT}...`);
  const dictContent = fs.readFileSync(SYSTEM_DICT, 'utf8');
  const allWords = dictContent
    .split('\n')
    .map(w => w.trim().toLowerCase())
    .filter(w => /^[a-z]{2,20}$/.test(w));

  const insertWordOnly = db.prepare(`
    INSERT OR IGNORE INTO dictionary (word, access_count) VALUES (?, 0)
  `);

  db.exec('BEGIN TRANSACTION');
  let addedCount = 0;
  for (const word of allWords) {
    const info = insertWordOnly.run(word);
    if (info.changes > 0) addedCount++;
  }
  db.exec('COMMIT');
  console.log(`Added ${addedCount} dictionary words without vectors.`);
}

const statsStmt = db.prepare('SELECT COUNT(*) as total, COUNT(vector) as embedded FROM dictionary');
const stats = statsStmt.get();
console.log(`\n=== Database Initialization Complete ===`);
console.log(`Total words: ${stats.total}`);
console.log(`Words with embeddings: ${stats.embedded}`);
