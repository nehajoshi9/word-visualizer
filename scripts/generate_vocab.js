import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CORE_COUNT = 8000;
const EXT_COUNT = 32000;
const TOTAL_COUNT = CORE_COUNT + EXT_COUNT; // 40,000 words
const SYSTEM_DICT = '/usr/share/dict/words';
const DIM = 384;
const BATCH_SIZE = 128;

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

  // 2. Supplement from system dictionary to reach exactly 40,000 words
  if (wordSet.size < TOTAL_COUNT && fs.existsSync(SYSTEM_DICT)) {
    const dictContent = fs.readFileSync(SYSTEM_DICT, 'utf8');
    const rawLines = dictContent.split('\n').map(l => l.trim().toLowerCase());
    const commonDictWords = Array.from(
      new Set(rawLines.filter(w => /^[a-z]{3,14}$/.test(w) && !wordSet.has(w)))
    );

    // Group candidates by first letter for balanced A-Z coverage
    const byLetter = {};
    for (let c = 97; c <= 122; c++) byLetter[String.fromCharCode(c)] = [];
    for (const w of commonDictWords) {
      if (byLetter[w[0]]) byLetter[w[0]].push(w);
    }

    let idx = 0;
    while (wordSet.size < TOTAL_COUNT) {
      let anyAdded = false;
      for (let c = 97; c <= 122; c++) {
        const letter = String.fromCharCode(c);
        const list = byLetter[letter];
        if (idx < list.length) {
          const w = list[idx];
          if (!wordSet.has(w)) {
            wordSet.add(w);
            anyAdded = true;
            if (wordSet.size >= TOTAL_COUNT) break;
          }
        }
      }
      if (!anyAdded) {
        // Fallback: fill sequentially if round-robin finishes
        for (const w of commonDictWords) {
          if (!wordSet.has(w)) {
            wordSet.add(w);
            if (wordSet.size >= TOTAL_COUNT) break;
          }
        }
        break;
      }
      idx++;
    }
  }

  const allWords = Array.from(wordSet).slice(0, TOTAL_COUNT);
  if (allWords.length !== TOTAL_COUNT) {
    throw new Error(`Failed to compile ${TOTAL_COUNT} words: got ${allWords.length}`);
  }

  const coreWords = allWords.slice(0, CORE_COUNT);
  const extWords = allWords.slice(CORE_COUNT, TOTAL_COUNT);

  console.log(`Vocabulary prepared: ${coreWords.length} core words, ${extWords.length} extended words (Total: ${allWords.length}).`);

  // 3. Embed all 40,000 words using all-MiniLM-L6-v2 and quantize to Int8
  console.log('Loading embedding pipeline (all-MiniLM-L6-v2)...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  });

  const coreInt8 = new Int8Array(CORE_COUNT * DIM);
  const extInt8 = new Int8Array(EXT_COUNT * DIM);

  const t0 = Date.now();
  console.log(`Beginning batch embedding & Int8 quantization (${TOTAL_COUNT} words in batches of ${BATCH_SIZE})...`);

  for (let i = 0; i < TOTAL_COUNT; i += BATCH_SIZE) {
    const batch = allWords.slice(i, i + BATCH_SIZE);
    const output = await extractor(batch, { pooling: 'mean', normalize: true });
    const data = output.data;

    for (let j = 0; j < batch.length; j++) {
      const wordIdx = i + j;
      if (wordIdx < CORE_COUNT) {
        const offset = wordIdx * DIM;
        for (let d = 0; d < DIM; d++) {
          const val = data[j * DIM + d];
          coreInt8[offset + d] = Math.round(Math.max(-1, Math.min(1, val)) * 127);
        }
      } else {
        const extIdx = wordIdx - CORE_COUNT;
        const offset = extIdx * DIM;
        for (let d = 0; d < DIM; d++) {
          const val = data[j * DIM + d];
          extInt8[offset + d] = Math.round(Math.max(-1, Math.min(1, val)) * 127);
        }
      }
    }

    const done = Math.min(i + BATCH_SIZE, TOTAL_COUNT);
    if (done % 4096 === 0 || done >= TOTAL_COUNT) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      const percent = Math.round((done / TOTAL_COUNT) * 100);
      console.log(`Embedded ${done} / ${TOTAL_COUNT} words (${percent}%) in ${elapsed}s...`);
    }
  }

  const publicDir = path.join(__dirname, '../public');
  if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
  }

  // 4. Save Tier 0 and Tier 1 versioned shards
  const coreJsonPath = path.join(publicDir, 'vectors-core-8k.v1.json');
  const coreBinPath = path.join(publicDir, 'vectors-core-8k.v1.bin');
  const extJsonPath = path.join(publicDir, 'vectors-ext-32k.v1.json');
  const extBinPath = path.join(publicDir, 'vectors-ext-32k.v1.bin');

  fs.writeFileSync(coreJsonPath, JSON.stringify(coreWords));
  fs.writeFileSync(coreBinPath, Buffer.from(coreInt8.buffer));
  fs.writeFileSync(extJsonPath, JSON.stringify(extWords));
  fs.writeFileSync(extBinPath, Buffer.from(extInt8.buffer));

  const totalTime = ((Date.now() - t0) / 1000).toFixed(1);
  console.log('\n=== Shard Generation Complete ===');
  console.log(`- Total Words: ${TOTAL_COUNT} (${coreWords.length} Core + ${extWords.length} Ext)`);
  console.log(`- Time: ${totalTime}s`);
  console.log(`- Tier 0 Core: ${coreJsonPath} (${(fs.statSync(coreJsonPath).size / 1024).toFixed(1)} KB)`);
  console.log(`- Tier 0 Binary: ${coreBinPath} (${(coreInt8.byteLength / (1024 * 1024)).toFixed(2)} MB, ${coreInt8.byteLength} bytes)`);
  console.log(`- Tier 1 Ext: ${extJsonPath} (${(fs.statSync(extJsonPath).size / 1024).toFixed(1)} KB)`);
  console.log(`- Tier 1 Binary: ${extBinPath} (${(extInt8.byteLength / (1024 * 1024)).toFixed(2)} MB, ${extInt8.byteLength} bytes)`);
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
