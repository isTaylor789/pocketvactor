"""Compara internal_batch_size 16/32/64 cargando el modelo directamente. 🧪

No imprime vectores. Uso: env/bin/python scripts/verify/bench_batch.py
"""

import statistics
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL = "./model"
DIM = 512
TEXTS = [f"bench document {i} " + "lorem ipsum dolor sit amet " * 8 for i in range(64)]


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL cuda no disponible")
        return 1
    t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL,
        trust_remote_code=True,
        truncate_dim=2048,
        device="cuda",
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    )
    load_s = time.perf_counter() - t0
    print(f"load_s={load_s:.1f} vram_mib={torch.cuda.memory_allocated() // 2**20}")

    fn = model.encode_document
    results: dict[int, tuple[float, float]] = {}
    vectors: dict[int, np.ndarray] = {}
    for bs in (16, 32, 64):
        # warmup
        fn(
            TEXTS[:2],
            batch_size=bs,
            precision="float32",
            normalize_embeddings=True,
            truncate_dim=DIM,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        times = []
        for _ in range(3):
            t = time.perf_counter()
            vecs = fn(
                TEXTS,
                batch_size=bs,
                precision="float32",
                normalize_embeddings=True,
                truncate_dim=DIM,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            times.append(time.perf_counter() - t)
        p50 = statistics.median(times)
        results[bs] = (p50, len(TEXTS) / p50)
        vectors[bs] = vecs
        print(f"batch={bs:2} p50_ms={p50 * 1000:7.1f} textos/s={len(TEXTS) / p50:7.1f}")

    base = vectors[32]
    for bs in (16, 64):
        diff = float(np.max(np.abs(vectors[bs].astype(np.float64) - base.astype(np.float64))))
        print(f"maxdiff batch={bs} vs 32: {diff:.6f}")

    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
