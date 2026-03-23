# attractors

Conversational attractor analysis for multi-turn LLM conversations. Embeds conversation turns, clusters them, and analyzes transition dynamics to identify conversational sinks.

## Pipeline

### 1. `embed.py`
Embeds all conversation turns using `nomic-embed-text-v1.5` on Modal (GPU).

- **Input:** `convo_v1_*.json` — conversation files, each a list of `{role, content}` turns
- **Output:** `convo_v1_*_embeddings.npy` — one per conversation, shape `[num_turns, 768]`

### 2. `cluster.py`
PCA dimensionality reduction + GMM clustering (diagonal covariance) with BIC model selection.

- **Input:** `convo_v1_*_embeddings.npy`
- **Output:**
  - `convo_v1_*_clusters.npy` — one per conversation, shape `[num_turns, 1]`
  - `all_projected.npy` — all turns projected to PCA space
  - `claude_cluster_diagnostics.png` — scree plot + BIC vs clusters

### 3. `transitions.py`
Builds empirical transition matrix and simulates cluster occupation from a uniform start.

- **Input:** `convo_v1_*_clusters.npy`
- **Output:**
  - `transition_matrix.npy` — shape `[n_clusters, n_clusters]`, row-normalized
  - `claude_transition_matrix.png` — raw transition heatmap + occupation bar chart after 30 steps

### 4. Top cluster sampling (inline script)
Extracts example turns from the highest-occupation clusters.

- **Input:** `transition_matrix.npy`, `convo_v1_*_clusters.npy`, `convo_v1_*.json`
- **Output:** `claude_top_cluster_samples.txt`

## Requirements

```
modal
numpy
matplotlib
scikit-learn
sentence-transformers
```
