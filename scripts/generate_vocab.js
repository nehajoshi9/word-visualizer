import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TARGET_COUNT = 30000;
const SYSTEM_DICT = '/usr/share/dict/words';

async function run() {
  const { pipeline, AutoTokenizer, env } = await import('@xenova/transformers');
  env.allowRemoteModels = false;

  console.log('Loading tokenizer from local cache...');
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
  const tokens = Object.values(tokenizer.model.vocab);

  // 1. Gather all whole alphabetic English words from MiniLM tokenizer (~21,700 words)
  // MiniLM tokens are naturally ordered by corpus frequency in Wikipedia/BookCorpus
  const minilmWords = tokens.filter(
    t => typeof t === 'string' && !t.startsWith('##') && /^[a-z]+$/.test(t)
  );

  const wordSet = new Set(minilmWords);
  console.log(`Gathered ${wordSet.size} whole words from MiniLM tokenizer.`);

  // 2. Supplement from system dictionary to reach 30,000 words
  if (wordSet.size < TARGET_COUNT && fs.existsSync(SYSTEM_DICT)) {
    const dictContent = fs.readFileSync(SYSTEM_DICT, 'utf8');
    const lines = dictContent.split('\n').map(l => l.trim());
    const commonDictWords = lines.filter(
      w => /^[a-z]{3,14}$/.test(w) && !wordSet.has(w)
    );

    // Group candidates by first letter for balanced A-Z coverage
    const byLetter = {};
    for (let c = 97; c <= 122; c++) byLetter[String.fromCharCode(c)] = [];
    for (const w of commonDictWords) {
      if (byLetter[w[0]]) byLetter[w[0]].push(w);
    }

    const needed = TARGET_COUNT - wordSet.size;
    const extraWords = [];
    let idx = 0;

    while (extraWords.length < needed) {
      let anyAdded = false;
      for (let c = 97; c <= 122; c++) {
        const letter = String.fromCharCode(c);
        const list = byLetter[letter];
        if (idx < list.length) {
          extraWords.push(list[idx]);
          anyAdded = true;
          if (extraWords.length >= needed) break;
        }
      }
      if (!anyAdded) break;
      idx++;
    }

    for (const w of extraWords) {
      wordSet.add(w);
    }
  }

  const targetWords = Array.from(wordSet).slice(0, TARGET_COUNT);
  console.log(`Target vocabulary prepared: ${targetWords.length} words.`);

  // 3. Embed all 30,000 words using all-MiniLM-L6-v2
  console.log('Loading embedding pipeline (all-MiniLM-L6-v2)...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  });

  const dim = 384;
  const total = targetWords.length;
  const floatBuffer = new Float32Array(total * dim);

  const BATCH_SIZE = 128;
  const t0 = Date.now();

  console.log(`Beginning batch embedding (${total} words in batches of ${BATCH_SIZE})...`);

  for (let i = 0; i < total; i += BATCH_SIZE) {
    const batch = targetWords.slice(i, i + BATCH_SIZE);
    const output = await extractor(batch, { pooling: 'mean', normalize: true });
    const data = output.data;

    for (let j = 0; j < batch.length; j++) {
      const offset = (i + j) * dim;
      for (let d = 0; d < dim; d++) {
        floatBuffer[offset + d] = data[j * dim + d];
      }
    }

    if ((i + BATCH_SIZE) % 2048 === 0 || i + BATCH_SIZE >= total) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      const done = Math.min(i + BATCH_SIZE, total);
      const percent = Math.round((done / total) * 100);
      console.log(`Embedded ${done} / ${total} words (${percent}%) in ${elapsed}s...`);
    }
  }

  const publicDir = path.join(__dirname, '../public');
  if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
  }

  // 4. Save word list and raw binary vector buffer
  fs.writeFileSync(path.join(publicDir, 'vocab_words.json'), JSON.stringify(targetWords));
  fs.writeFileSync(path.join(publicDir, 'vocab_vectors.bin'), Buffer.from(floatBuffer.buffer));

  const totalTime = ((Date.now() - t0) / 1000).toFixed(1);
  console.log('\n=== Generation Complete ===');
  console.log(`- Words: ${targetWords.length}`);
  console.log(`- Time: ${totalTime}s`);
  console.log(`- Output words file: public/vocab_words.json (${(fs.statSync(path.join(publicDir, 'vocab_words.json')).size / 1024).toFixed(1)} KB)`);
  console.log(`- Output vector file: public/vocab_vectors.bin (${(floatBuffer.byteLength / (1024 * 1024)).toFixed(2)} MB)`);
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
