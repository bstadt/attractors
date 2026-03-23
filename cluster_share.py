"""
Visualize cluster occupation share over conversation time.
For each turn index t, compute the fraction of conversations in each cluster.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path(__file__).parent

# Load cluster shards
cluster_files = sorted(glob.glob(str(data_dir / "convo_v1_*_clusters.npy")))
shards = [np.load(f).flatten() for f in cluster_files]

n_clusters = max(s.max() for s in shards) + 1
max_turns = max(len(s) for s in shards)
print(f"{len(shards)} conversations, {n_clusters} clusters, max {max_turns} turns")

# Build occupation matrix: [max_turns, n_clusters]
# For each turn t, count how many conversations are in each cluster
occupation = np.zeros((max_turns, n_clusters))
active_convos = np.zeros(max_turns)

for shard in shards:
    for t, c in enumerate(shard):
        occupation[t, c] += 1
        active_convos[t] += 1

# Normalize to shares
active_convos[active_convos == 0] = 1
shares = occupation / active_convos[:, None]

# Find clusters that ever exceed a threshold to keep the plot readable
peak_share = shares.max(axis=0)
top_k = 20
top_clusters = np.argsort(peak_share)[-top_k:][::-1]

# Stacked area plot for top clusters
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

# Top panel: stacked area of top clusters
top_shares = shares[:, top_clusters]
ax1.stackplot(range(max_turns), top_shares.T, labels=[f"C{c}" for c in top_clusters], alpha=0.8)
ax1.set_ylabel("Share of conversations")
ax1.set_title(f"Cluster Occupation Share Over Time (top {top_k} clusters)")
ax1.set_xlim(0, max_turns - 1)
ax1.set_ylim(0, 1)
ax1.legend(loc="upper right", ncol=4, fontsize=7)

# Bottom panel: number of active conversations at each turn
ax2.fill_between(range(max_turns), active_convos, alpha=0.5)
ax2.set_xlabel("Turn")
ax2.set_ylabel("Active conversations")
ax2.set_xlim(0, max_turns - 1)

plt.tight_layout()
plt.savefig(data_dir / "cluster_share_over_time.png", dpi=150)
print("Saved cluster_share_over_time.png")
