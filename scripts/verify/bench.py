"""Benchmark de latencia/throughput contra server vivo. No imprime vectores. 🧪

Uso: BASE_URL=http://127.0.0.1:9009 scripts/verify/bench.py
"""

import os
import statistics
import time

import httpx

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:9009")
RESULTS: list[tuple[str, list[float], int, list[int]]] = []


def pct(values: list[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[k]


def run_case(client: httpx.Client, name: str, count: int, dim: int, reps: int) -> bool:
    walls: list[float] = []
    server_ms: list[int] = []
    tokens: list[int] = []
    for rep in range(reps):
        texts = [f"bench {name} {rep} item {i}" for i in range(count)]
        payload = {"texts": texts, "truncate_dim": dim}
        t0 = time.perf_counter()
        r = client.post(f"{BASE}/embed/query", json=payload)
        walls.append((time.perf_counter() - t0) * 1000)
        body = r.json()
        if r.status_code != 200:
            print(f"FAIL {name}: http={r.status_code} {body.get('message')}")
            return False
        if body["data"]["count"] != count or body["data"]["dim"] != dim:
            print(f"FAIL {name}: contrato inesperado")
            return False
        server_ms.append(int(body["meta"]["ms"]))
        tokens.append(int(body["meta"]["tokens"]))
    RESULTS.append((name, walls, count, server_ms))
    return True


def main() -> int:
    client = httpx.Client(timeout=180)
    ready = client.get(f"{BASE}/ready")
    if ready.status_code != 200:
        print(f"FAIL ready http={ready.status_code}")
        return 1

    # cold vs warm (el modelo ya está cargado; esto mide warmup de kernels/rutas)
    cold: list[float] = []
    for i in range(3):
        t0 = time.perf_counter()
        r = client.post(f"{BASE}/embed/query", json={"texts": [f"warmup {i}"], "truncate_dim": 512})
        cold.append((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            print(f"FAIL warmup http={r.status_code}")
            return 1

    cases = [
        ("count=1 dim=512", 1, 512, 10),
        ("count=8 dim=512", 8, 512, 5),
        ("count=32 dim=512", 32, 512, 3),
        ("count=64 dim=512", 64, 512, 2),
        ("count=8 dim=256", 8, 256, 3),
        ("count=8 dim=1024", 8, 1024, 3),
        ("count=8 dim=2048", 8, 2048, 3),
    ]
    for name, count, dim, reps in cases:
        if not run_case(client, name, count, dim, reps):
            return 1

    print(f"warmup_ms={[round(x, 1) for x in cold]}\n")
    print(f"{'caso':20} {'reps':>4} {'p50_ms':>8} {'p95_ms':>8} {'srv_p50':>8} {'textos/s':>9}")
    for name, walls, count, server_ms in RESULTS:
        total_s = sum(walls) / 1000
        throughput = (count * len(walls)) / total_s if total_s else 0
        print(
            f"{name:20} {len(walls):>4} {statistics.median(walls):>8.1f} "
            f"{pct(walls, 95):>8.1f} {statistics.median(server_ms):>8.0f} {throughput:>9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
