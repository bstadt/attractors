"""
Compute empirical transition matrix P(cluster_{t+1} | cluster_t) from cluster shards.
Display raw transition matrix and 10-iteration power iteration side by side.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path(__file__).parent

# Load cluster shards
cluster_files = sorted(glob.glob(str(data_dir / "convo_v1_*_clusters.npy")))
print(f"Loading {len(cluster_files)} cluster shards...")

shards = [np.load(f).flatten() for f in cluster_files]

n_clusters = max(s.max() for s in shards) + 1
print(f"Number of clusters: {n_clusters}")

# Count transitions
counts = np.zeros((n_clusters, n_clusters), dtype=np.float64)
for shard in shards:
    for t in range(len(shard) - 1):
        counts[shard[t], shard[t + 1]] += 1

# Row-normalize
row_sums = counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
transition = counts / row_sums

# Simulate uniform distribution forward
n_steps = 30
p = np.ones(n_clusters) / n_clusters
history = [p.copy()]
for _ in range(n_steps):
    p = p @ transition
    history.append(p.copy())
history = np.array(history)  # [n_steps+1, n_clusters]

# Save
np.save(data_dir / "transition_matrix.npy", transition)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle("Claude. Transition Dynamics", fontsize=16, fontweight="bold", y=1.02)

im1 = ax1.imshow(transition, cmap="Reds", aspect="auto", interpolation="nearest")
ax1.set_xlabel("Cluster t+1")
ax1.set_ylabel("Cluster t")
ax1.set_title("Raw Transition Matrix")
plt.colorbar(im1, ax=ax1, label="P(transition)", shrink=0.8)

# Bar chart of cluster occupation after n_steps
ax2.bar(range(n_clusters), history[-1], width=1.0, color="steelblue")
ax2.axhline(1.0 / n_clusters, color="r", linestyle="--", alpha=0.5, label="Uniform baseline")
ax2.set_xlabel("Cluster")
ax2.set_ylabel("Occupation probability")
ax2.set_title("Simulated Cluster Occupation After 30 Steps")
ax2.legend()

plt.tight_layout()
plt.savefig(data_dir / "claude_transition_matrix.png", dpi=150, bbox_inches="tight")
print("Saved transition_matrix.png")
