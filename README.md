# Semantic Vector Space Visualizer

An interactive web application that converts any English words to high-dimensional semantic vectors (384-D), compresses them to 2D via **Principal Component Analysis (PCA)**, plots them as interactive nodes on a coordinate plane, and allows you to select any combination of words to discover semantically similar English words from across the language.

---

## Architecture: SQLite DB + 8,000-Word LRU Cache

The system uses an **unrestricted English language architecture**:

```
[User Input / Multi-Word Selection]
                │
                ▼
      ┌──────────────────┐      Cache Hit (0 ms)
      │    LRU Cache     │ ────────────────────────► Instant Vector Return
      │ (8,000 Hot Words)│
      └─────────┬────────┘
                │ Cache Miss
                ▼
      ┌──────────────────┐      Found in DB
      │ SQLite Database  │ ────────────────────────► Promote to LRU & Return
      │  (236,623 words) │
      └─────────┬────────┘
                │ Not yet embedded
                ▼
      ┌──────────────────┐
      │ Neural Embedder  │ ────► Persist to SQLite ──► Push to LRU ──► Return
      │ (MiniLM-L6-v2)   │
      └──────────────────┘
```

1. **SQLite Database (`words.db`)**:
   - Stores the entire English dictionary (**236,623 words** from system dictionary).
   - Zero length restrictions: short 1-letter words (`"a"`, `"i"`) to long complex terms (`"antidisestablishmentarianism"`).
   - Persists all embedded vectors as raw 384-D `Float32Array` BLOBs so computations are saved permanently.

2. **LRU Cache (Capacity: 8,000 Words)**:
   - **Initial State**: Pre-populated on startup with the **8,000 most frequently used English words** with pre-computed vectors.
   - Operates in $O(1)$ time for both lookups and updates.
   - As new or rare words are requested, they are promoted into the cache while cold entries are naturally evicted to disk.
   - Semantic similarity queries evaluate directly against the active 8,000 words in the cache in < 2 ms!

3. **Deterministic PCA 2D Dimensionality Reduction**:
   - Computes the top two principal components ($PC_1, PC_2$) across all active canvas words.
   - Preserves global semantic distances: related words cluster naturally, while unrelated words separate.
   - Live variance indicators show the percentage of semantic information retained.

4. **Multi-Word Semantic Search & Centroid**:
   - Click words on the plane to select/deselect them.
   - Visual **Semantic Centroid** marker ($\odot$) and dashed lines link selected words.
   - Joint semantic scoring:
     $$\text{Score}(w) = 0.55 \cdot \left(\frac{1}{|S|} \sum_{s \in S} \cos(v_s, v_w)\right) + 0.45 \cdot \left(\min_{s \in S} \cos(v_s, v_w)\right)$$
   - 1-Click **`+ Add`** button to plot recommended words onto the plane.

---

## Running the Application

1. **Start the SQLite + LRU Cache Backend**:
   ```bash
   npm run server
   ```
   *Runs on port 3001 and preloads the 8,000 most frequent English words into the LRU Cache.*

2. **Start the Frontend UI**:
   ```bash
   npm run dev
   ```
   *Open [http://localhost:5173](http://localhost:5173) in your browser.*
