"""
PCA + GMM (diag) clustering pipeline for conversation turn embeddings.

Step 1: PCA on all embeddings, select elbow dimension (12).
Step 2: GMM (diag) clustering in PCA space, BIC selection over [2..512].
Produces per-conversation cluster shards and a single two-panel diagnostic plot.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

data_dir = Path(__file__).parent

# Load all embedding shards
emb_files = sorted(glob.glob(str(data_dir / "convo_v1_*_embeddings.npy")))
print(f"Loading {len(emb_files)} embedding shards...")

shards = [np.load(f) for f in emb_files]
all_embeddings = np.concatenate(shards, axis=0)
print(f"Total embeddings: {all_embeddings.shape}")

# --- Step 1: PCA ---
max_components = min(200, all_embeddings.shape[0], all_embeddings.shape[1])
print(f"Fitting PCA with {max_components} components...")
pca = PCA(n_components=max_components, random_state=42)
pca.fit(all_embeddings)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
elbow_dim = 12
print(f"Elbow at {elbow_dim} components (cumulative variance: {cumulative[elbow_dim - 1]:.3f})")

# --- Step 2: Project and cluster ---
print(f"Projecting to {elbow_dim} dimensions...")
all_projected = pca.transform(all_embeddings)[:, :elbow_dim]
np.save(data_dir / "all_projected.npy", all_projected)

ks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
bics = []

for k in ks:
    print(f"Fitting k={k}...", end=" ", flush=True)
    gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42, max_iter=300, reg_covar=1e-4)
    gmm.fit(all_projected)
    bic = gmm.bic(all_projected)
    bics.append(bic)
    print(f"BIC={bic:.0f}")

best_idx = np.argmin(bics)
best_k = ks[best_idx]
print(f"\nBest k={best_k} (BIC={bics[best_idx]:.0f})")

# --- Single two-panel plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Claude. Clustering Diagnostics", fontsize=16, fontweight="bold", y=1.02)

ax1.plot(range(1, max_components + 1), pca.explained_variance_, ".-", markersize=3)
ax1.axvline(elbow_dim, color="r", linestyle="--", label=f"Elbow={elbow_dim}")
ax1.set_xlabel("Component")
ax1.set_ylabel("Eigenvalue")
ax1.set_title("Scree Plot")
ax1.legend()

ax2.plot(ks, bics, "o-")
ax2.axvline(best_k, color="r", linestyle="--", label=f"Best k={best_k}")
ax2.set_xscale("log", base=2)
ax2.set_xlabel("Number of clusters")
ax2.set_ylabel("BIC")
ax2.set_title(f"BIC vs Clusters (Diag GMM, {elbow_dim}D PCA)")
ax2.legend()

plt.tight_layout()
plt.savefig(data_dir / "claude_cluster_diagnostics.png", dpi=150, bbox_inches="tight")
print("Saved cluster_diagnostics.png")

# --- Refit best and assign ---
print(f"Refitting best model (k={best_k})...")
best_gmm = GaussianMixture(n_components=best_k, covariance_type="diag", random_state=42, max_iter=300, reg_covar=1e-4)
best_gmm.fit(all_projected)
all_labels = best_gmm.predict(all_projected)

offset = 0
for i, shard in enumerate(shards):
    n = shard.shape[0]
    labels = all_labels[offset:offset + n].reshape(-1, 1)
    out_path = emb_files[i].replace("_embeddings.npy", "_clusters.npy")
    np.save(out_path, labels)
    offset += n

print(f"Saved {len(shards)} cluster shards")
