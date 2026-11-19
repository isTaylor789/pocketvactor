"""Robustez: timeout, concurrencia sin cola, rate-limit y repetición. 🧪"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

Q = "/embed/query"
BASE = {"texts": ["hola"], "truncate_dim": 512}


def _app(**kwargs):
    app = create_app(Settings(skip_model_load=True, **kwargs))
    svc = app.state.svc
    svc.model = object()  # ready sin GPU
    svc._token_stats = lambda texts, kind: (10, [False] * len(texts))  # noqa: E731
    return app, svc


def _zeros(*args, **kwargs):
    texts = args[1]
    return np.zeros((len(texts), 512), dtype=np.float32)


def test_inference_timeout_returns_504():
    app, svc = _app(inference_timeout=0.01)

    def slow(*args, **kwargs):
        time.sleep(0.3)
        return _zeros(*args, **kwargs)

    svc._encode_sync = slow  # type: ignore[method-assign]
    r = TestClient(app).post(Q, json=BASE)
    body = r.json()
    assert r.status_code == 504, body
    assert body["code"] == 504 and body["error"]["type"] == "gpu_timeout"
    assert body["error"]["details"][0]["reason"] == "inference_timeout"


def test_many_concurrent_requests_rejected_while_gpu_busy():
    app, svc = _app()
    entered = threading.Event()
    release = threading.Event()

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=15)
        return _zeros(*args, **kwargs)

    svc._encode_sync = slow  # type: ignore[method-assign]
    client = TestClient(app)
    first: dict = {}

    def first_call():
        first["r"] = client.post(Q, json={"texts": ["primero"], "truncate_dim": 512})

    t = threading.Thread(target=first_call)
    t.start()
    assert entered.wait(timeout=10)

    def second(i):
        return client.post(Q, json={"texts": [f"segundo {i}"], "truncate_dim": 512})

    with ThreadPoolExecutor(max_workers=10) as ex:
        responses = list(ex.map(second, range(10)))

    for r in responses:
        body = r.json()
        assert r.status_code == 503, body
        assert body["code"] == 503 and body["error"]["type"] == "gpu_busy"

    release.set()
    t.join(timeout=15)
    assert first["r"].status_code == 200


def test_rate_limit_429_and_system_endpoints_free():
    app, svc = _app(rate_limit_per_minute=2)
    svc._encode_sync = _zeros  # type: ignore[method-assign]
    client = TestClient(app)

    assert [client.post(Q, json=BASE).status_code for _ in range(2)] == [200, 200]
    r = client.post(Q, json=BASE)
    body = r.json()
    assert r.status_code == 429, body
    assert body["code"] == 429 and body["error"]["type"] == "rate_limited"
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_repeated_requests_leave_no_broken_state():
    app, svc = _app()
    svc._encode_sync = _zeros  # type: ignore[method-assign]
    client = TestClient(app)
    codes = [client.post(Q, json=BASE).status_code for _ in range(30)]
    assert codes == [200] * 30
