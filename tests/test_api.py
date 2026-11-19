"""Tests de validación y sobre — sin cargar el modelo (SKIP=1). 🧪"""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

client = TestClient(create_app(Settings(skip_model_load=True)))
Q = "/embed/query"
BASE = {"truncate_dim": 512}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "success"


def test_info_lists_capabilities():
    r = client.get("/info")
    body = r.json()
    assert body["data"]["truncate_dims"] == [2048, 1024, 512, 256]
    assert "POST /embed/query" in body["data"]["endpoints"]


def test_missing_truncate_dim():
    r = client.post(Q, json={"texts": ["hola"]})
    body = r.json()
    assert r.status_code == 422
    assert body["status"] == "error"
    # ausente cae en literal_error → se reporta como truncate_dim inválido/ausente
    assert body["error"]["details"][0]["field"] == "truncate_dim"


def test_empty_list():
    r = client.post(Q, json={"texts": [], **BASE})
    assert r.json()["error"]["details"][0]["reason"] == "empty_list"


def test_whitespace_and_empty_report_all_indices():
    r = client.post(Q, json={"texts": ["ok", "   ", ""], **BASE})
    reasons = {d["index"]: d["reason"] for d in r.json()["error"]["details"]}
    assert reasons == {1: "whitespace_only", 2: "empty"}


def test_too_many_and_too_long():
    r = client.post(Q, json={"texts": ["x"] * 65, **BASE})
    d = r.json()["error"]["details"][0]
    assert d["reason"] == "too_many_texts" and d["received"] == 65 and d["max"] == 64
    r = client.post(Q, json={"texts": ["a" * 32001], **BASE})
    d = r.json()["error"]["details"][0]
    assert d["reason"] == "exceeds_max_chars" and d["received_chars"] == 32001


def test_not_string_and_null():
    r = client.post(Q, json={"texts": ["ok", 123, None], **BASE})
    reasons = {d["index"]: d["reason"] for d in r.json()["error"]["details"]}
    assert reasons[1] == "not_string" and reasons[2] == "null_item"


def test_invalid_params():
    r = client.post(Q, json={"texts": ["ok"], "truncate_dim": 100})
    assert r.json()["error"]["type"] == "invalid_param"
    assert r.json()["error"]["details"][0]["received"] == 100
    r = client.post(Q, json={"texts": ["ok"], "truncate_dim": 512, "precision": "int4"})
    assert r.json()["error"]["details"][0]["reason"] == "invalid_precision"
    assert r.json()["error"]["details"][0]["received"] == "int4"
    r = client.post(Q, json={"texts": ["ok"], "truncate_dim": 512, "normalize": "true"})
    assert r.json()["error"]["details"][0]["reason"] == "invalid_normalize"


def test_unsupported_precisions_rejected():
    """int8/uint8 no tienen soporte upstream: deben dar 422, no 500."""
    for precision in ("int8", "uint8"):
        r = client.post(Q, json={"texts": ["ok"], "truncate_dim": 512, "precision": precision})
        body = r.json()
        assert r.status_code == 422, body
        assert body["error"]["details"][0]["reason"] == "invalid_precision"
        assert body["error"]["details"][0]["allowed"] == ["float32", "binary", "ubinary"]


def test_valid_without_model_is_503_not_500():
    """Validación pasa → como no hay modelo, 503 controlado (no 500).
    SPEC-errors §1: code == HTTP status real, nunca 200 con error dentro."""
    r = client.post(Q, json={"texts": ["hola"], **BASE})
    body = r.json()
    assert r.status_code == 503
    assert body["status"] == "error" and body["code"] == 503
    assert body["error"]["type"] == "not_ready"


def test_ready_without_model_is_503():
    r = client.get("/ready")
    body = r.json()
    assert r.status_code == 503
    assert body["code"] == 503 and body["error"]["type"] == "not_ready"


def test_envelope_meta_has_endpoint():
    r = client.post(Q, json={"texts": [], **BASE})
    assert r.json()["meta"]["endpoint"] == "query"
    r = client.post("/embed/document", json={"texts": [], **BASE})
    assert r.json()["meta"]["endpoint"] == "document"


def test_every_detail_carries_message():
    """Todo detail del DTO sale con {field, reason, message} — sin asumir nada."""
    cases = [
        {"texts": [], "truncate_dim": 512},
        {"texts": ["ok", None], "truncate_dim": 512},
        {"texts": ["ok"], "truncate_dim": 100},
        {"texts": ["ok"], "truncate_dim": 512, "precision": "int4"},
        {"texts": ["ok"], "truncate_dim": 512, "normalize": "true"},
        {"texts": ["hola"]},
    ]
    for payload in cases:
        r = client.post(Q, json=payload)
        body = r.json()
        assert body["status"] == "error"
        assert "message" in body and len(body["message"]) > 0
        for d in body["error"]["details"]:
            assert {"field", "reason", "message"} <= set(d), d
            assert len(d["message"]) > 0, d


def test_wrong_content_type_has_friendly_message():
    r = client.post(
        Q, content='{"texts":["hola"],"truncate_dim":512}', headers={"Content-Type": "text/plain"}
    )
    body = r.json()
    assert r.status_code == 422
    d = body["error"]["details"][0]
    assert d["reason"] == "invalid_body"
    assert "application/json" in d["message"]


def test_malformed_json_has_friendly_message():
    r = client.post(Q, content="not json{{{", headers={"Content-Type": "application/json"})
    body = r.json()
    assert r.status_code == 422
    d = body["error"]["details"][0]
    assert d["reason"] == "invalid_json"
    assert len(d["message"]) > 0


def test_gpu_busy_second_request_rejected_immediately():
    """Semáforo 1 sin cola: 2do concurrente → 503 al momento, no espera. 💥"""
    import threading
    import time

    import numpy as np

    from app.main import create_app

    app2 = create_app(Settings(skip_model_load=True))
    c2 = TestClient(app2)
    svc = app2.state.svc
    svc.model = object()  # ready sin GPU
    svc._token_stats = lambda texts, kind: (10, [False] * len(texts))  # noqa: E731
    entered = threading.Event()
    release = threading.Event()

    def slow(*a, **k):
        entered.set()
        assert release.wait(timeout=15)
        return np.zeros((1, 512), dtype=np.float32)

    svc._encode_sync = slow  # type: ignore[method-assign]
    res1: dict = {}

    def first():
        res1["r"] = c2.post(Q, json={"texts": ["hola"], "truncate_dim": 512})

    t = threading.Thread(target=first)
    t.start()
    assert entered.wait(timeout=15)
    time.sleep(0.3)  # el lock sigue tomado por el 1ro
    t0 = time.perf_counter()
    r2 = c2.post(Q, json={"texts": ["otro"], "truncate_dim": 512})
    dt = time.perf_counter() - t0
    body2 = r2.json()
    assert r2.status_code == 503, body2
    assert body2["code"] == 503 and body2["error"]["type"] == "gpu_busy"
    assert len(body2["error"]["details"][0]["message"]) > 0
    assert dt < 5, f"debió rechazar al momento, tardó {dt:.1f}s (¿hay cola?)"  # sin FIFO
    release.set()
    t.join(timeout=15)
    assert res1["r"].status_code == 200


def test_rate_limit_soft_429_with_envelope():
    """Rate-limit suave: pasado el tope → 429 con sobre y message. 🐢"""
    from app.main import create_app

    app3 = create_app(Settings(skip_model_load=True, rate_limit_per_minute=3))
    c3 = TestClient(app3)
    codes = [c3.post(Q, json={"texts": ["hola"], **BASE}).status_code for _ in range(4)]
    assert codes[:3] == [503, 503, 503]  # pasan el limit, pegan en not_ready
    assert codes[3] == 429
    body = c3.post(Q, json={"texts": ["hola"], **BASE}).json()
    assert body["code"] == 429 and body["error"]["type"] == "rate_limited"
    assert len(body["error"]["details"][0]["message"]) > 0
    # sistema libre del limit
    assert c3.get("/health").status_code == 200
