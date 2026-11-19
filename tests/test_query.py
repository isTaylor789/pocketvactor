"""Contrato de POST /embed/query (sin GPU). 🔍"""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

client = TestClient(create_app(Settings(skip_model_load=True)))
Q = "/embed/query"
BASE = {"texts": ["Which planet is known as the Red Planet?"], "truncate_dim": 512}


def test_valid_query_reaches_not_ready():
    r = client.post(Q, json=BASE)
    body = r.json()
    assert r.status_code == 503
    assert body["code"] == 503 and body["error"]["type"] == "not_ready"
    assert body["meta"]["endpoint"] == "query"


def test_query_requires_truncate_dim():
    r = client.post(Q, json={"texts": ["hola"]})
    body = r.json()
    assert r.status_code == 422
    assert body["error"]["details"][0]["field"] == "truncate_dim"


def test_query_rejects_truncate_dim_coercion():
    for value in ("512", 512.0, True, None):
        r = client.post(Q, json={"texts": ["ok"], "truncate_dim": value})
        body = r.json()
        assert r.status_code == 422, body
        assert body["error"]["details"][0]["reason"] == "invalid_truncate_dim"
