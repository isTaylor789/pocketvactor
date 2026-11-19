"""Mapeo de errores de validación Pydantic → sobre {status:error}. 🧾"""

import json
from typing import Any

from fastapi.exceptions import RequestValidationError

from .schemas import _MULTI_MARKER


def _short(value: Any, limit: int = 80) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


def validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        typ = err.get("type", "")
        ctx = err.get("ctx", {}) or {}
        # loc típica: ('body', 'texts', 3) | ('body', 'truncate_dim') | ('body',)
        parts = [p for p in loc if p != "body"]

        # 1️⃣ Nuestro validador multi-índice (todos los textos malos a la vez)
        # El DTO ya trae {field, reason, message}; se pasan tal cual.
        if typ == "value_error" and isinstance(ctx.get("error"), ValueError):
            msg = str(ctx["error"])
            if msg.startswith(_MULTI_MARKER):
                for d in json.loads(msg[len(_MULTI_MARKER) :]):
                    d.setdefault(
                        "message", f"{d.get('field', 'texts')}: {d.get('reason', 'invalid')}"
                    )
                    details.append(d)
                continue

        # reconstrucción simple y legible:
        if not parts:
            field = "body"
        elif parts[0] == "texts" and len(parts) == 1:
            field = "texts"
        elif parts[0] == "texts" and len(parts) >= 2 and isinstance(parts[1], int):
            field = f"texts[{parts[1]}]"
        else:
            field = ".".join(str(p) for p in parts)
        index = (
            parts[1]
            if len(parts) >= 2 and parts[0] == "texts" and isinstance(parts[1], int)
            else None
        )

        # 2️⃣ Mapeo tipo Pydantic → reason máquina + message humano.
        # Todo detail sale con {field, reason, message}; el resto es contexto.
        reason = typ
        message = f"{field}: {typ}, revisa el body"
        extra: dict[str, Any] = {}
        if typ == "json_invalid":
            reason, field = "invalid_json", "body"
            message = "body inválido, manda JSON válido"
            extra = {"hint": "revisa comas/comillas del JSON"}
        elif typ == "model_attributes_type":
            reason, field = "invalid_body", "body"
            message = "usa Content-Type: application/json con un objeto {texts, truncate_dim}"
            extra = {"hint": "usa Content-Type: application/json"}
        elif parts and parts[0] == "texts" and typ == "too_short":
            reason, field = "empty_list", "texts"
            message = "manda al menos 1 texto"
        elif parts and parts[0] == "texts" and typ == "too_long":
            reason, field = "too_many_texts", "texts"
            raw = err.get("input", [])
            received = len(raw) if isinstance(raw, list) else "unknown"
            maximum = (err.get("ctx") or {}).get("max_length", 64)
            message = f"máximo {maximum} textos por request (recibidos: {received})"
            extra = {"received": received, "max": maximum}
        elif typ == "list_type":
            reason = "not_a_list"
            message = "texts debe ser lista, incluso para 1 texto"
            extra = {"hint": "aunque sea 1 texto, mándalo como lista 📝"}
        elif typ == "string_type" and index is not None:
            reason = "not_string"
            message = f"{field} debe ser string, envuélvelo en comillas"
            extra = {"hint": "envuélvelo en comillas"}
        elif parts and parts[-1] == "truncate_dim":
            reason = "invalid_truncate_dim"
            message = (
                "truncate_dim debe ser 2048, 1024, 512 o 256 (es requerido, sin default oculto)"
            )
            extra = {"allowed": [2048, 1024, 512, 256], "received": _short(err.get("input"))}
            field, index = "truncate_dim", None
        elif parts and parts[-1] == "precision":
            reason = "invalid_precision"
            message = "precision debe ser float32, binary o ubinary (case-sensitive)"
            extra = {
                "allowed": ["float32", "binary", "ubinary"],
                "received": _short(err.get("input")),
            }
            field, index = "precision", None
        elif parts and parts[-1] == "normalize":
            reason = "invalid_normalize"
            message = "normalize debe ser boolean true/false, no string"
            extra = {"hint": 'usa boolean JSON true/false, no "true" string 🪤'}
            field, index = "normalize", None
        elif typ == "missing":
            reason = "missing"
            message = f"falta campo requerido: {field} (truncate_dim es requerido ⚠️)"
            extra = {"hint": "truncate_dim es requerido ⚠️"}

        details.append(
            {
                "field": field,
                "reason": reason,
                "message": message,
                **({"index": index} if index is not None else {}),
                **extra,
            }
        )
    return details


def error_type_for(details: list[dict[str, Any]]) -> str:
    reasons = {d["reason"] for d in details}
    if reasons & {"invalid_truncate_dim", "invalid_precision", "invalid_normalize"}:
        return "invalid_param"
    return "validation_error"
