"""Verificación numérica contra server vivo. Solo imprime PASS/FAIL y métricas. 🧪

No imprime vectores: usa hashes, normas, shapes y diferencias máximas.
Uso: BASE_URL=http://127.0.0.1:9009 scripts/verify/verify_numeric.py
"""

import hashlib
import os

import httpx
import numpy as np

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:9009")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))


def call(
    kind: str,
    texts: list[str],
    dim: int = 512,
    precision: str = "float32",
    normalize: bool = True,
) -> dict:
    payload = {
        "texts": texts,
        "truncate_dim": dim,
        "precision": precision,
        "normalize": normalize,
    }
    r = httpx.post(f"{BASE}/embed/{kind}", json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"{kind} {precision}: {r.status_code} {r.text[:200]}")
    return r.json()


def vecs(body: dict) -> np.ndarray:
    return np.asarray(body["data"]["embeddings"])


def digest(v: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(v, dtype=np.float32).tobytes()).hexdigest()[:16]


def maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main() -> int:
    # 1. Determinismo (mismo request dos veces)
    a = call("query", ["determinism probe"])
    b = call("query", ["determinism probe"])
    check(
        "determinismo query",
        digest(vecs(a)[0]) == digest(vecs(b)[0]),
        f"hash={digest(vecs(a)[0])} maxdiff={maxdiff(vecs(a), vecs(b)):.3g}",
    )

    a = call("document", ["determinism probe"])
    b = call("document", ["determinism probe"])
    check(
        "determinismo document",
        digest(vecs(a)[0]) == digest(vecs(b)[0]),
        f"hash={digest(vecs(a)[0])}",
    )

    # 2. Independencia del batch
    solo = vecs(call("query", ["alpha"]))[0]
    batch = vecs(call("query", ["x", "alpha", "y"]))[1]
    d = maxdiff(solo, batch)
    check("batch independiente", d < 1e-5, f"maxdiff={d:.3g}")

    # 3. Orden garantizado
    pa = vecs(call("query", ["b", "a"]))
    vb = vecs(call("query", ["b"]))[0]
    va = vecs(call("query", ["a"]))[0]
    check(
        "orden embeddings[i] ↔ texts[i]",
        maxdiff(pa[0], vb) < 1e-5 and maxdiff(pa[1], va) < 1e-5,
        f"d0={maxdiff(pa[0], vb):.3g} d1={maxdiff(pa[1], va):.3g}",
    )

    # 4. Normalización
    v = vecs(call("query", ["norm probe"], normalize=True))[0]
    n = float(np.linalg.norm(v))
    check("normalize=true → ‖v‖≈1", abs(n - 1) < 1e-3, f"norm={n:.6f}")

    v2 = vecs(call("query", ["norm probe"], normalize=False))[0]
    n2 = float(np.linalg.norm(v2))
    check("normalize=false → vector finito", np.isfinite(v2).all(), f"norm={n2:.6f}")

    # 5. Dimensiones MRL
    for dim in (2048, 1024, 512, 256):
        body = call("query", ["dim probe"], dim=dim)
        arr = vecs(body)
        ok = (
            body["data"]["dim"] == dim
            and body["data"]["count"] == 1
            and arr.shape == (1, dim)
            and np.isfinite(arr).all()
        )
        check(f"dim={dim}", ok, f"shape={arr.shape}")

    # 6. Precisions binarias (empaquetadas)
    for precision in ("binary", "ubinary"):
        body = call("query", ["binary probe"], dim=512, precision=precision)
        arr = vecs(body)
        expected = (1, 512 // 8)
        lo, hi = (-128, 127) if precision == "binary" else (0, 255)
        ok = (
            arr.shape == expected
            and int(arr.min()) >= lo
            and int(arr.max()) <= hi
            and np.any(arr != 0)
        )
        check(
            f"precision={precision}",
            ok,
            f"shape={arr.shape} min={int(arr.min())} max={int(arr.max())}",
        )

    # 7. Query y document usan prompts distintos
    q = vecs(call("query", ["same text"]))[0]
    dvec = vecs(call("document", ["same text"]))[0]
    diff = maxdiff(q, dvec)
    check("query ≠ document", diff > 1e-6, f"maxdiff={diff:.3g}")

    # 8. Duplicados se embedden 2 veces y son idénticos
    dup = vecs(call("query", ["hola", "hola"]))
    dd = maxdiff(dup[0], dup[1])
    check("duplicados idénticos", dd < 1e-6, f"maxdiff={dd:.3g}")

    # 9. Límite exacto de 8000 chars pasa
    body = call("query", ["a" * 8000])
    check(
        "8000 chars (límite inclusivo)",
        body["data"]["count"] == 1 and vecs(body).shape == (1, 512),
        f"shape={vecs(body).shape}",
    )

    # 10. Unicode / emoji / RTL
    body = call("query", ["Marte 🔴 المريخ 火星", "🔴🔴🔴"])
    check(
        "unicode/emoji/RTL",
        body["data"]["count"] == 2 and vecs(body).shape == (2, 512),
        f"shape={vecs(body).shape}",
    )

    # 11. Metadatos de truncado bien formados
    body = call("query", ["meta probe"])
    meta = body["meta"]
    check(
        "meta.truncated/meta.tokens",
        isinstance(meta.get("truncated"), list)
        and len(meta["truncated"]) == 1
        and isinstance(meta["truncated"][0], bool)
        and isinstance(meta.get("tokens"), int)
        and meta["tokens"] > 0,
        f"tokens={meta.get('tokens')} truncated={meta.get('truncated')}",
    )

    # 12. Todos los vectores finitos
    allv = np.concatenate(
        [vecs(call("query", ["finite probe"])), vecs(call("document", ["finite probe"]))]
    )
    check("sin NaN/Inf", bool(np.isfinite(allv).all()))

    failed = [r for r in RESULTS if not r[1]]
    print(f"\nRESULT PASS={len(RESULTS) - len(failed)} FAIL={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
