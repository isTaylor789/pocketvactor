"""Hammer de concurrencia contra server vivo: 1 procesa, el resto 503 sin cola. 🧪

No imprime vectores. Uso: BASE_URL=http://127.0.0.1:9009 scripts/verify/verify_concurrency.py
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:9009")
Q = f"{BASE}/embed/query"
LONG = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 130)[:8000]
FIRST_PAYLOAD = {"texts": [LONG] * 64, "truncate_dim": 2048}


def main() -> int:
    first: dict = {}

    def first_call():
        first["r"] = httpx.post(Q, json=FIRST_PAYLOAD, timeout=180)

    t = threading.Thread(target=first_call)
    t.start()
    time.sleep(0.5)  # el 1ro debe estar ocupando la GPU

    def second(i: int):
        payload = {"texts": [f"segundo {i}"], "truncate_dim": 512}
        return httpx.post(Q, json=payload, timeout=30)

    with ThreadPoolExecutor(max_workers=10) as ex:
        responses = list(ex.map(second, range(10)))

    t.join(timeout=180)
    r1 = first.get("r")
    codes = [r.status_code for r in responses]
    types = [r.json().get("error", {}).get("type") for r in responses]
    ok_seconds = all(c == 503 and t == "gpu_busy" for c, t in zip(codes, types, strict=False))
    ok_first = r1 is not None and r1.status_code == 200

    print(f"primero http={getattr(r1, 'status_code', None)}")
    print(f"concurrentes códigos={codes}")
    print(f"concurrentes tipos={types}")
    print(f"{'PASS' if ok_seconds else 'FAIL'} concurrentes 503 gpu_busy")
    print(f"{'PASS' if ok_first else 'FAIL'} primero completa 200")
    print(f"{'PASS' if 500 not in codes else 'FAIL'} sin 500")
    return 0 if ok_seconds and ok_first and 500 not in codes else 1


if __name__ == "__main__":
    raise SystemExit(main())
