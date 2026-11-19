"""Edge cases de la SPEC (sin GPU): validación, unicidad y contrato. 🧪"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import EmbedRequest

client = TestClient(create_app(Settings(skip_model_load=True)))
Q = "/embed/query"
D = "/embed/document"
DIM = {"truncate_dim": 512}


def _post(payload, path=Q):
    return client.post(path, json=payload)


@pytest.mark.parametrize(
    ("payload", "field", "reason"),
    [
        ({"truncate_dim": 512}, "texts", "missing"),
        ({"texts": "hola", **DIM}, "texts", "not_a_list"),
        ({"texts": {}, **DIM}, "texts", "not_a_list"),
        ({"texts": [["hola"]], **DIM}, "texts[0]", "not_string"),
        ({"texts": [{"text": "hola"}], **DIM}, "texts[0]", "not_string"),
        ({"texts": [True], **DIM}, "texts[0]", "not_string"),
        ({"texts": [0], **DIM}, "texts[0]", "not_string"),
        ({"texts": [123456789], **DIM}, "texts[0]", "not_string"),
        ({"texts": ["a" * 32001], **DIM}, "texts[0]", "exceeds_max_chars"),
        ({"texts": ["ok"], "truncate_dim": "512"}, "truncate_dim", "invalid_truncate_dim"),
        ({"texts": ["ok"], "truncate_dim": 512.0}, "truncate_dim", "invalid_truncate_dim"),
        ({"texts": ["ok"], "truncate_dim": True}, "truncate_dim", "invalid_truncate_dim"),
        ({"texts": ["ok"], "truncate_dim": None}, "truncate_dim", "invalid_truncate_dim"),
        ({"texts": ["ok"], **DIM, "normalize": 1}, "normalize", "invalid_normalize"),
        ({"texts": ["ok"], **DIM, "normalize": "true"}, "normalize", "invalid_normalize"),
        ({"texts": ["ok"], **DIM, "normalize": None}, "normalize", "invalid_normalize"),
    ],
)
def test_invalid_edge_cases(payload, field, reason):
    r = _post(payload)
    body = r.json()
    assert r.status_code == 422, body
    assert body["status"] == "error" and body["code"] == 422
    details = body["error"]["details"]
    assert any(d["field"] == field and d["reason"] == reason for d in details), details
    for d in details:
        assert {"field", "reason", "message"} <= set(d)
        assert len(d["message"]) > 0


def test_missing_texts_and_dim_report_both():
    r = _post({})
    body = r.json()
    assert r.status_code == 422
    reasons = {(d["field"], d["reason"]) for d in body["error"]["details"]}
    assert ("texts", "missing") in reasons
    assert ("truncate_dim", "invalid_truncate_dim") in reasons


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": ["a" * 32000], **DIM},  # límite inclusivo
        {"texts": ["Marte 🔴 المريخ 火星"], **DIM},
        {"texts": ["🔴🔴🔴"], **DIM},
        {"texts": ["hola", "hola"], **DIM},
        {"texts": ["ok"], **DIM, "foo": 1},  # extra ignorado
        {"texts": ["ok"], **DIM, "normalize": False},
        {"texts": ["ok"], **DIM, "precision": "binary"},
        {"texts": ["ok"], **DIM, "precision": "ubinary"},
    ],
)
def test_valid_edge_cases_reach_not_ready(payload):
    """Validación pasa; sin modelo el resultado controlado es 503, nunca 422/500."""
    r = _post(payload)
    body = r.json()
    assert r.status_code == 503, body
    assert body["status"] == "error" and body["code"] == 503
    assert body["error"]["type"] == "not_ready"


def test_invalid_edge_cases_apply_to_document():
    r = _post({"texts": ["ok", None], **DIM}, path=D)
    body = r.json()
    assert r.status_code == 422
    details = {d["index"]: d["reason"] for d in body["error"]["details"]}
    assert details == {1: "null_item"}
    assert body["meta"]["endpoint"] == "document"


def test_info_lists_only_supported_precisions():
    body = client.get("/info").json()
    assert body["data"]["precisions"] == ["float32", "binary", "ubinary"]


def test_limits_come_from_settings():
    """max_texts/max_chars deben respetar Settings, no estar hardcodeados."""
    old = (EmbedRequest._max_texts, EmbedRequest._max_chars)
    try:
        app = create_app(Settings(skip_model_load=True, max_texts=2, max_chars=5))
        c = TestClient(app)

        r = c.post(Q, json={"texts": ["a", "b", "c"], "truncate_dim": 512})
        d = r.json()["error"]["details"][0]
        assert r.status_code == 422
        assert d["reason"] == "too_many_texts" and d["received"] == 3 and d["max"] == 2

        r = c.post(Q, json={"texts": ["123456"], "truncate_dim": 512})
        d = r.json()["error"]["details"][0]
        assert r.status_code == 422
        assert d["reason"] == "exceeds_max_chars"
        assert d["received_chars"] == 6 and d["max_chars"] == 5
    finally:
        EmbedRequest.configure(*old)
