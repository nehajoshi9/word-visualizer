# Word Visualizer

> An interactive semantic vector space explorer running entirely in the browser.

[![Live Demo](https://img.shields.io/badge/demo-online-brightgreen.svg)](https://nehajoshi9.github.io/word-visualizer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Word Visualizer** is a client-side NLP playground that maps words into dense semantic vector space (384-D), dynamically projects them onto an interactive 2D coordinate plane via **Principal Component Analysis (PCA)**, and calculates joint semantic intersections across the English language.

Explore the live application at **[https://nehajoshi9.github.io/word-visualizer/](https://nehajoshi9.github.io/word-visualizer/)**.

---

## Overview

Traditional dictionaries and thesauruses operate on discrete, 1-to-1 lookups—you search for a single word to find its synonyms.

**Word Visualizer** treats language as continuous geometry. Every word is transformed into a high-dimensional mathematical coordinate. When multiple words are placed on the plane, the system calculates their shared geometric centroid and queries an in-memory vocabulary to find words that sit at the conceptual intersection of all selected terms.

---

## Key Capabilities

- **Interactive 2D Semantic Plane**: A minimal, responsive canvas where physical distance corresponds to conceptual similarity.
- **Real-Time PCA Projection**: Computes the top two principal components ($PC_1, PC_2$) of the active word set on the fly, preserving relative distances without coordinate jitter.
- **Multi-Word Semantic Intersection**: Select any combination of words (e.g., `apple` + `orange`, or `computer` + `brain`) to discover terms that share high semantic affinity with all of them simultaneously.
- **Pairwise Distance Inspection**: Hover over the connection between any two selected words to view their high-dimensional cosine similarity and semantic vector distance.
- **Zero-Server Architecture**: Runs 100% in the browser via WebAssembly (ONNX Runtime). No external APIs, no backend server, and no cloud database dependencies.

---

## How It Works

### 1. Semantic Embedding
Words are converted into 384-dimensional dense vectors using a quantized `all-MiniLM-L6-v2` Sentence Transformer. Words with related meanings point in similar directions in vector space.

### 2. Dual Gram Matrix PCA
High-dimensional vectors are projected onto 2D coordinates using an $O(N^2)$ Dual Gram Matrix formulation of Principal Component Analysis:

1. **Mean Centering**: $X_c = X - \mu$
2. **Kernel Gram Matrix**: $K = X_c X_c^T$
3. **Eigendecomposition**: Power iteration with deflation extracts the two dominant eigenvectors ($u_1, u_2$) and eigenvalues ($\lambda_1, \lambda_2$).
4. **2D Mapping**: $z_i = \left( u_{1,i}\sqrt{\lambda_1},\; u_{2,i}\sqrt{\lambda_2} \right)$

### 3. Joint Multi-Word Affinity
When a subset of words $S = \{s_1, \dots, s_k\}$ is selected, candidate vocabulary words $w$ are evaluated using a balanced metric that balances average semantic similarity with worst-case distance:

$$\text{Affinity}(w) = 0.55 \cdot \left(\frac{1}{|S|} \sum_{s \in S} \cos(v_s, v_w)\right) + 0.45 \cdot \left(\min_{s \in S} \cos(v_s, v_w)\right)$$

This ensures that recommended words are relevant to *all* active selections rather than skewing toward just one.

---

## Architecture

Word Visualizer uses a **fully automatic, zero-UI 3-layer sharded vocabulary architecture** with `Int8` quantization and `IndexedDB` persistence:

- **Layer 1 — Core 8k Shard (`vectors-core-8k.v1.*`)**: Top 8,000 most frequent English words quantized to signed `Int8` (~3.07 MB binary buffer). Fetched immediately on boot so the 2D plane and semantic search are interactive right away.
- **Layer 2 — Silent Idle 32k Upgrade (`vectors-ext-32k.v1.*`)**: Words ranked 8,001 to 40,000 quantized to signed `Int8` (~12.29 MB binary buffer). Fetched in the background via `requestIdleCallback` and saved to `IndexedDB` (`word-visualizer-shards`). Subsequent visits load all 40,000 words in ~40 ms with zero network calls.
- **Layer 3 — In-Browser WASM Fallback**: Any novel, rare, or domain-specific word outside the 40,000 pre-computed vocabulary is embedded on the fly in the browser via `@xenova/transformers` (WebAssembly) and dynamically added to the active candidate pool.
- **Sub-10ms Cosine Similarity Scans**: Direct dot products over contiguous `Int8Array` buffers evaluate 40,000 words in $< 10\text{ ms}$ in browser RAM.
- **Local Server (Optional)**: Includes an optional Node.js + SQLite backend (`server.js`) indexed with 240,000+ words from the full dictionary for local experimentation.

---

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or newer)
- npm

### Installation & Local Run

```bash
# Clone the repository
git clone https://github.com/nehajoshi9/word-visualizer.git
cd word-visualizer

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Building for Production

```bash
npm run build
```

The output in `dist/` is ready to deploy to any static hosting provider (GitHub Pages, Vercel, Netlify, Cloudflare Pages).

---

## Tech Stack

- **Frontend**: React 19, TypeScript, Tailwind CSS, Vite
- **Machine Learning**: `@xenova/transformers` (ONNX Runtime WebAssembly), Sentence Transformers
- **Math & Algorithms**: Custom Dual Gram Matrix PCA with Rayleigh quotient iteration
- **Storage / Indexing**: Sharded `Int8Array` binary buffers, IndexedDB caching, unified $O(1)$ index, optional `node:sqlite`

---

## License

This project is licensed under the [MIT License](LICENSE).
