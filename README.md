# Word Visualizer

An interactive semantic vector space visualizer. Type any English words to map them into high-dimensional semantic space (384-D), compress them to 2D via **Principal Component Analysis (PCA)**, and explore multi-word semantic intersections—running **100% client-side** with zero backend or database required.

**Live Demo**: [https://nehajoshi9.github.io/word-visualizer/](https://nehajoshi9.github.io/word-visualizer/)

---

## Why Is This Useful?

1. **Multi-Word Semantic Intersection (Beyond a Thesaurus)**:
   A standard thesaurus only gives synonyms for a single word. Word Visualizer allows you to select *any combination of words* (e.g., `apple` + `orange`, or `king` + `queen`, or `computer` + `brain`) and discover concepts that sit at the joint mathematical intersection of all of them.
2. **Intuitive Spatial Geometry for NLP & Machine Learning**:
   Translates abstract mathematical concepts—dense embeddings, cosine distance, and PCA projection—into an interactive physical plane where geometric distance reflects conceptual meaning.
3. **Completely Serverless & Free**:
   Runs 384-dimensional neural network inference, typed array vector math, and an 8,000-word index directly inside your browser via WebAssembly (ONNX Runtime). No API keys, no server costs, and no Supabase required.

---

## Features

- **Any English Word**: Type any word and press `Enter` to embed and plot it.
- **2D PCA Projection**: Dynamically computes the top two principal components ($PC_1, PC_2$) across all active words in real time.
- **Multi-Word Selection**: Click nodes on the plane or in the sidebar to select/deselect them.
- **Semantic Similarity Discovery**: Queries the top 8,000 most frequent English words to find concepts with the highest joint affinity to all selected words.
- **Pairwise Distance Tooltip**: Hover over any line connecting selected words to inspect:
  - **Similarity**: Cosine similarity percentage.
  - **Semantic distance**: $d_{\cos} = 1 - \cos(\theta)$ in 384-D space.
  - **Plane distance**: Euclidean distance on the 2D projected plane in pixels.
- **Right-Click to Delete**: Right-click any node to delete it from the plane and recompute the layout.
- **Keyboard Shortcuts**:
  - `Enter`: Add word and automatically select it.
  - `Esc`: Instantly deselect all words.
  - Always-focused input: start typing anytime without clicking the text box.
- **Isolated Plane Zoom**: Trackpad pinch and mouse wheel zoom strictly within the plane without zooming the browser page.
- **Auto-Fit View**: Automatically zooms out to fit new words with comfortable margin, plus a 1-click fit icon button (`⛶`).

---

## How It Works

```
                                          ┌──────────────────────┐
[User Types Word] ──► 384-D Embedding ──► │  Dual Gram Matrix    │ ──► Projected 2D Plane
                                          │  PCA Decomposition   │     (All nodes same color)
                                          └──────────────────────┘
                                                     ▲
                                                     │ Selected Words
                                                     ▼
                                          ┌──────────────────────┐
                                          │ Joint Similarity     │ ──► Ranked Similar Words
                                          │ Across 8,000-Word    │     (Click to add)
                                          │ In-Memory LRU Cache  │
                                          └──────────────────────┘
```

### 1. High-Dimensional Embeddings
Every word is mapped into a 384-dimensional vector space using `all-MiniLM-L6-v2`. Words with similar semantic meanings have vectors pointing in similar angular directions.

### 2. Principal Component Analysis (PCA)
To project high-dimensional vectors onto a 2D plane without jitter:
1. Centering: $X_c = X - \mu$
2. Gram Matrix: $K = X_c X_c^T$
3. Eigenvalue decomposition using power iteration with deflation computes the top two eigenvectors ($u_1, u_2$) and eigenvalues ($\lambda_1, \lambda_2$).
4. 2D Coordinates: $z_i = \left( u_{1,i}\sqrt{\lambda_1},\; u_{2,i}\sqrt{\lambda_2} \right)$.

### 3. Multi-Word Semantic Similarity
When a set of words $S = \{s_1, \dots, s_k\}$ is selected, candidate words $w$ from the active 8,000-word vocabulary are scored using a composite function that rewards high average similarity while penalizing candidates that are distant from any single selected word:

$$\text{Score}(w) = 0.55 \cdot \left(\frac{1}{|S|} \sum_{s \in S} \cos(v_s, v_w)\right) + 0.45 \cdot \left(\min_{s \in S} \cos(v_s, v_w)\right)$$

---

## How Does It Run Without a Database on GitHub Pages?

GitHub Pages is a static file host that does not run backend servers or databases. Word Visualizer functions completely serverless:

1. **Pre-computed Vector Asset (`vocab_vectors.bin`)**:
   An 11 MB binary buffer of normalized `Float32Array` embeddings for the top 8,000 most frequent English words is hosted statically and downloaded once by the browser.
2. **In-Memory Browser LRU Cache**:
   The vectors live in browser RAM. Multi-word cosine similarity queries execute over all 8,000 words in under 2 milliseconds.
3. **Client-Side Neural Network via WebAssembly**:
   If you type an obscure or novel word outside the top 8,000, `@xenova/transformers` (running in ONNX WebAssembly) generates the vector on the fly directly inside your browser.

*(An optional local SQLite backend (`server.js`) containing 236,623 words from the full dictionary is also included if you ever want to run a local backend).*

---

## Development

```bash
# Clone repository
git clone https://github.com/nehajoshi9/word-visualizer.git
cd word-visualizer

# Install dependencies
npm install

# Start local development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Deploy to GitHub Pages

```bash
npm run deploy
```

---

## License

MIT
