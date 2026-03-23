"""
Embed conversation turns from JSON shards using nomic-embed-text-v1.5 on Modal.

For each convo_v1_XXX.json, produces convo_v1_XXX_embeddings.npy with shape [num_turns, 768].
"""

import json
import glob
import numpy as np
from pathlib import Path
from typing import List

import modal

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "d802ae16c9caed4d197895d27c6d529434cd8c6d"

image = modal.Image.debian_slim().pip_install(
    "torch==2.6.0",
    "sentence-transformers==3.4.1",
    "einops==0.8.1",
    "numpy",
)

app = modal.App("attractors-embedding", image=image)

CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    gpu="H100",
    volumes={CACHE_DIR: cache_vol},
    timeout=60 * 30,
    scaledown_window=60 * 10,
    max_containers=10,
)
class EmbeddingModel:
    @modal.enter()
    def setup(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_folder=CACHE_DIR,
            trust_remote_code=True,
        )

    @modal.method()
    def embed_batch(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


def load_turns(json_path: str) -> List[str]:
    with open(json_path) as f:
        turns = json.load(f)
    return [t["content"] for t in turns]


def main():
    data_dir = Path(__file__).parent
    json_files = sorted(glob.glob(str(data_dir / "convo_v1_*.json")))
    print(f"Found {len(json_files)} conversation files")

    # Collect all turns with file boundaries
    all_texts = []
    file_ranges = []  # (start_idx, end_idx, json_path)
    for jf in json_files:
        turns = load_turns(jf)
        start = len(all_texts)
        all_texts.extend(turns)
        file_ranges.append((start, len(all_texts), jf))

    print(f"Total turns to embed: {len(all_texts)}")

    # Embed everything in one shot via Modal
    CHUNK_SIZE = 1000
    chunks = [all_texts[i:i + CHUNK_SIZE] for i in range(0, len(all_texts), CHUNK_SIZE)]
    print(f"Submitting {len(chunks)} chunks to Modal...")

    with app.run():
        model = EmbeddingModel()
        futures = [model.embed_batch.spawn(chunk) for chunk in chunks]
        results = [f.get() for f in futures]

    all_embeddings = np.concatenate(results, axis=0)
    print(f"Embeddings shape: {all_embeddings.shape}")

    # Slice back into per-conversation shards
    for start, end, jf in file_ranges:
        shard = all_embeddings[start:end]
        out_path = jf.replace(".json", "_embeddings.npy")
        np.save(out_path, shard)

    print(f"Saved {len(file_ranges)} embedding shards")


if __name__ == "__main__":
    main()
