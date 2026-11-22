import { DatabaseSync } from 'node:sqlite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DB_PATH = path.join(__dirname, '../words.db');
const SYSTEM_DICT = '/usr/share/dict/words';
const VOCAB_WORDS_PATH = path.join(__dirname, '../public/vocab_words.json');
const VOCAB_VECTORS_PATH = path.join(__dirname, '../public/vocab_vectors.bin');

console.log('Initializing words.db with full English dictionary and top 8,000 frequent words...');

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

// 2. Load top 8,000 frequent words with pre-computed vectors
const topWords = JSON.parse(fs.readFileSync(VOCAB_WORDS_PATH, 'utf8'));
const vectorBuffer = fs.readFileSync(VOCAB_VECTORS_PATH);
const dim = 384;
const byteStride = dim * 4;

console.log(`Loading ${topWords.length} frequent words with vectors into SQLite...`);

const insertWordWithVector = db.prepare(`
  INSERT INTO dictionary (word, vector, freq_rank, access_count, last_accessed)
  VALUES (?, ?, ?, 1, ?)
  ON CONFLICT(word) DO UPDATE SET
    vector = excluded.vector,
    freq_rank = excluded.freq_rank,
    last_accessed = excluded.last_accessed
`);

const now = Date.now();
db.exec('BEGIN TRANSACTION');
for (let i = 0; i < topWords.length; i++) {
  const w = topWords[i].toLowerCase();
  const subBuf = vectorBuffer.subarray(i * byteStride, (i + 1) * byteStride);
  insertWordWithVector.run(w, subBuf, i + 1, now);
}
db.exec('COMMIT');

console.log(`Top ${topWords.length} frequent words committed to SQLite.`);

// 3. Load remaining English words from system dictionary
if (fs.existsSync(SYSTEM_DICT)) {
  console.log(`Loading remaining English words from ${SYSTEM_DICT}...`);
  const dictContent = fs.readFileSync(SYSTEM_DICT, 'utf8');
  const allWords = dictContent
    .split('\n')
    .map(w => w.trim().toLowerCase())
    .filter(w => w.length > 0 && /^[a-z]+$/.test(w));

  const uniqueAllWords = Array.from(new Set(allWords));
  console.log(`Found ${uniqueAllWords.length} total unique words in system dictionary.`);

  const insertWordOnly = db.prepare(`
    INSERT OR IGNORE INTO dictionary (word, vector, freq_rank, access_count, last_accessed)
    VALUES (?, NULL, NULL, 0, NULL)
  `);

  db.exec('BEGIN TRANSACTION');
  let inserted = 0;
  for (const w of uniqueAllWords) {
    insertWordOnly.run(w);
    inserted++;
  }
  db.exec('COMMIT');

  console.log(`All English words added to SQLite. Total words in dictionary: ${uniqueAllWords.length}.`);
}

// Check counts
const countRow = db.prepare('SELECT COUNT(*) as total, COUNT(vector) as with_vector FROM dictionary').get();
console.log(`Database initialized successfully! Total words: ${countRow.total}, with vectors: ${countRow.with_vector}`);

db.close();
