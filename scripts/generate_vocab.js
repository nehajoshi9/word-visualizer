import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function run() {
  const { pipeline, AutoTokenizer, env } = await import('@xenova/transformers');
  env.allowRemoteModels = false;

  console.log('Loading tokenizer & pipeline from local cache...');
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
  const tokens = Object.values(tokenizer.model.vocab);

  // Filter for clean alphabetic English words of reasonable length
  // Exclude subwords (##) and non-alphabetic tokens
  const rawWords = tokens.filter(
    t => typeof t === 'string' && !t.startsWith('##') && /^[a-z]{2,16}$/.test(t)
  );

  // Deduplicate and filter out trivial 2-letter tokens except common ones
  const allowedTwoLetter = new Set(['in', 'on', 'at', 'by', 'to', 'up', 'so', 'no', 'go', 'we', 'he', 'me', 'my', 'do', 'it', 'is', 'as', 'if', 'or']);
  const cleanWords = rawWords.filter(w => w.length > 2 || allowedTwoLetter.has(w));

  // Take top 8,000 most frequent English words covering all general usage
  const targetWords = cleanWords.slice(0, 8000);
  console.log(`Selected ${targetWords.length} general English vocabulary words.`);

  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { quantized: true });
  const dim = 384;
  const total = targetWords.length;
  const floatBuffer = new Float32Array(total * dim);

  const BATCH_SIZE = 128;
  const t0 = Date.now();

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

    if ((i + BATCH_SIZE) % 1024 === 0 || i + BATCH_SIZE >= total) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`Embedded ${Math.min(i + BATCH_SIZE, total)} / ${total} words (${elapsed}s)...`);
    }
  }

  const publicDir = path.join(__dirname, '../public');
  if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
  }

  fs.writeFileSync(path.join(publicDir, 'vocab_words.json'), JSON.stringify(targetWords));
  fs.writeFileSync(path.join(publicDir, 'vocab_vectors.bin'), Buffer.from(floatBuffer.buffer));

  console.log('Finished successfully:');
  console.log(`- public/vocab_words.json: ${targetWords.length} words`);
  console.log(`- public/vocab_vectors.bin: ${(floatBuffer.byteLength / (1024 * 1024)).toFixed(2)} MB`);
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
