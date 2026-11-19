"""Contrato de POST /embed/document (sin GPU). 📄"""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

client = TestClient(create_app(Settings(skip_model_load=True)))
D = "/embed/document"
BASE = {
    "texts": ["Mars, known for its reddish appearance, is referred to as the Red Planet."],
    "truncate_dim": 512,
}


def test_valid_document_reaches_not_ready():
    r = client.post(D, json=BASE)
    body = r.json()
    assert r.status_code == 503
    assert body["code"] == 503 and body["error"]["type"] == "not_ready"
    assert body["meta"]["endpoint"] == "document"


def test_document_reuses_query_validation():
    r = client.post(D, json={"texts": ["ok", None], "truncate_dim": 512})
    body = r.json()
    assert r.status_code == 422
    assert {d["index"]: d["reason"] for d in body["error"]["details"]} == {1: "null_item"}


def test_document_rejects_unsupported_precision():
    r = client.post(D, json={"texts": ["ok"], "truncate_dim": 512, "precision": "int8"})
    body = r.json()
    assert r.status_code == 422
    assert body["error"]["details"][0]["reason"] == "invalid_precision"
