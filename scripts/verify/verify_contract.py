"""Validación de contrato contra server vivo. Solo resúmenes, sin vectores. 🧪"""

import os
from pathlib import Path

import httpx

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:9009")
Q = f"{BASE}/embed/query"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))


def enum_of(prop: dict):
    if "enum" in prop:
        return prop["enum"]
    for key in ("anyOf", "oneOf"):
        for sub in prop.get(key, []):
            if "enum" in sub:
                return sub["enum"]
    return None


def error_reason(body: dict) -> str:
    return body.get("error", {}).get("details", [{}])[0].get("reason", "")


def main() -> int:
    client = httpx.Client(timeout=180)

    # --- documentación / OpenAPI ---
    docs = client.get(f"{BASE}/docs")
    check("/docs responde 200", docs.status_code == 200, f"http={docs.status_code}")
    openapi = client.get(f"{BASE}/openapi.json")
    spec = openapi.json()
    paths = set(spec.get("paths", {}))
    expected = {"/health", "/ready", "/info", "/embed/query", "/embed/document"}
    check("/openapi.json tiene todos los paths", openapi.status_code == 200 and expected <= paths)

    schema = spec["components"]["schemas"]["EmbedRequest"]
    required = set(schema.get("required", []))
    props = schema["properties"]
    check("truncate_dim requerido", "truncate_dim" in required)
    check(
        "truncate_dim enum correcto",
        enum_of(props["truncate_dim"]) == [2048, 1024, 512, 256],
        str(enum_of(props["truncate_dim"])),
    )
    check(
        "precision enum correcto",
        enum_of(props["precision"]) == ["float32", "binary", "ubinary"],
        str(enum_of(props["precision"])),
    )
    check("texts minItems=1", props["texts"].get("minItems") == 1)

    # --- /info ---
    info = client.get(f"{BASE}/info").json()
    data = info["data"]
    check(
        "info límites/precisions",
        data["limits"] == {"max_texts": 64, "max_chars": 8000, "max_tokens": 32768}
        and data["truncate_dims"] == [2048, 1024, 512, 256]
        and data["precisions"] == ["float32", "binary", "ubinary"],
    )

    # --- success envelopes ---
    for path, endpoint in (("/embed/query", "query"), ("/embed/document", "document")):
        r = client.post(f"{BASE}{path}", json={"texts": ["contract probe"], "truncate_dim": 512})
        body = r.json()
        ok = (
            r.status_code == 200
            and body.get("status") == "success"
            and body.get("code") == 200
            and "data" in body
            and "error" not in body
            and body["meta"].get("endpoint") == endpoint
            and set(body) == {"status", "code", "message", "data", "meta"}
        )
        check(f"success envelope {endpoint}", ok, f"http={r.status_code}")

    # --- error matrix: code == HTTP + sobre + detail ---
    json_h = {"Content-Type": "application/json"}
    error_cases = [
        ("empty_list", Q, {"texts": [], "truncate_dim": 512}, json_h),
        ("missing", Q, {"truncate_dim": 512}, json_h),
        ("not_a_list", Q, {"texts": "hola", "truncate_dim": 512}, json_h),
        ("too_many_texts", Q, {"texts": ["x"] * 65, "truncate_dim": 512}, json_h),
        ("exceeds_max_chars", Q, {"texts": ["a" * 8001], "truncate_dim": 512}, json_h),
        ("invalid_truncate_dim", Q, {"texts": ["ok"], "truncate_dim": 100}, json_h),
        ("invalid_truncate_dim_float", Q, {"texts": ["ok"], "truncate_dim": 512.0}, json_h),
        (
            "invalid_precision",
            Q,
            {"texts": ["ok"], "truncate_dim": 512, "precision": "int8"},
            json_h,
        ),
        (
            "invalid_normalize",
            Q,
            {"texts": ["ok"], "truncate_dim": 512, "normalize": "true"},
            json_h,
        ),
        ("null_item", Q, {"texts": [None], "truncate_dim": 512}, json_h),
        ("not_string", Q, {"texts": [123], "truncate_dim": 512}, json_h),
        ("empty", Q, {"texts": [""], "truncate_dim": 512}, json_h),
        ("whitespace_only", Q, {"texts": ["   "], "truncate_dim": 512}, json_h),
        ("invalid_body", Q, {"texts": ["ok"], "truncate_dim": 512}, {"Content-Type": "text/plain"}),
    ]
    emitted: set[str] = set()
    for name, url, payload, headers in error_cases:
        if headers.get("Content-Type") == "text/plain":
            r = client.post(url, content='{"texts":["ok"]}', headers=headers)
        else:
            r = client.post(url, json=payload)
        body = r.json()
        det = body.get("error", {}).get("details", [{}])[0]
        reason = error_reason(body)
        emitted.add(reason)
        ok = (
            r.status_code == 422
            and body.get("status") == "error"
            and body.get("code") == r.status_code
            and "error" in body
            and "data" not in body
            and {"field", "reason", "message"} <= set(det)
        )
        check(f"error {name} ({reason})", ok, f"http={r.status_code}")

    r = client.post(Q, content="not json{{{", headers={"Content-Type": "application/json"})
    body = r.json()
    emitted.add(error_reason(body))
    check(
        "error malformed json (invalid_json)",
        r.status_code == 422 and error_reason(body) == "invalid_json" and body["code"] == 422,
        f"http={r.status_code}",
    )

    # --- catálogo SPEC-errors §5 ---
    text = Path("docs/SPEC-errors.md").read_text()
    section = text[text.index("## 5.") :]
    catalog_line = next(line for line in section.splitlines() if line.startswith("`"))
    catalog = {item.strip().split(" (")[0] for item in catalog_line.strip("`").split("|")}
    missing = sorted(emitted - catalog)
    check("reasons emitidos están en SPEC-errors §5", not missing, f"faltan={missing}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\nRESULT PASS={len(RESULTS) - len(failed)} FAIL={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
