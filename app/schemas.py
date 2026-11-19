"""Schemas de request/response — sobre {status, code, message, data|error, meta}. 📦"""

import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

TruncateDim = Literal[2048, 1024, 512, 256]
# int8/uint8 no tienen soporte upstream en voyage-4-nano: no se aceptan.
Precision = Literal["float32", "binary", "ubinary"]

#: Marcador interno: el validador empaqueta TODOS los índices malos en un ValueError.
_MULTI_MARKER = "__MULTI__"


class EmbedRequest(BaseModel):
    _max_texts: ClassVar[int] = 64
    _max_chars: ClassVar[int] = 32000

    texts: list[Any] = Field(min_length=1)
    truncate_dim: TruncateDim  # requerido, sin default mágico ⚠️
    precision: Precision = "float32"
    normalize: StrictBool = True

    @classmethod
    def configure(cls, max_texts: int, max_chars: int) -> None:
        """Sincroniza los límites con Settings sin hardcodearlos en el DTO."""
        cls._max_texts = max_texts
        cls._max_chars = max_chars

    @field_validator("truncate_dim", mode="before")
    @classmethod
    def _strict_truncate_dim(cls, v: Any) -> Any:
        # Sin coerción: 512.0, "512", true y null deben fallar (SPEC edge #14).
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"truncate_dim inválido: {v!r}")
        return v

    @field_validator("texts")
    @classmethod
    def _check_texts(cls, v: list[Any]) -> list[str]:
        bad: list[dict[str, Any]] = []
        if len(v) > cls._max_texts:
            raise ValueError(
                _MULTI_MARKER
                + json.dumps(
                    [
                        {
                            "field": "texts",
                            "reason": "too_many_texts",
                            "message": f"máximo {cls._max_texts} textos por request "
                            f"(recibidos: {len(v)})",
                            "received": len(v),
                            "max": cls._max_texts,
                        }
                    ]
                )
            )
        for i, t in enumerate(v):
            field = f"texts[{i}]"
            if t is None:
                bad.append(
                    {
                        "index": i,
                        "field": field,
                        "reason": "null_item",
                        "message": f"{field} es null, manda un string",
                    }
                )
            elif not isinstance(t, str):
                bad.append(
                    {
                        "index": i,
                        "field": field,
                        "reason": "not_string",
                        "message": f"{field} debe ser string, envuélvelo en comillas",
                        "received_type": type(t).__name__,
                    }
                )
            elif t == "":
                bad.append(
                    {
                        "index": i,
                        "field": field,
                        "reason": "empty",
                        "message": f"{field} está vacío, manda al menos 1 caracter",
                    }
                )
            elif t.strip() == "":
                bad.append(
                    {
                        "index": i,
                        "field": field,
                        "reason": "whitespace_only",
                        "message": f"{field} solo tiene espacios, el modelo devolvería basura",
                        "received_preview": t[:80],
                    }
                )
            elif len(t) > cls._max_chars:
                bad.append(
                    {
                        "index": i,
                        "field": field,
                        "reason": "exceeds_max_chars",
                        "message": f"{field} excede {cls._max_chars} chars ({len(t)}), "
                        "pártelo o resume",
                        "received_chars": len(t),
                        "max_chars": cls._max_chars,
                    }
                )
        if bad:
            raise ValueError(_MULTI_MARKER + json.dumps(bad))
        return v


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="allow")  # permite received_chars, allowed, etc.

    field: str
    reason: str
    message: str  # mensaje humano por detalle, siempre presente
    index: int | None = None


class ErrorBody(BaseModel):
    type: str
    details: list[ErrorDetail] = []


class ErrorEnvelope(BaseModel):
    status: Literal["error"] = "error"
    code: int
    message: str
    error: ErrorBody
    meta: dict[str, Any] = {}


class SuccessEnvelope(BaseModel):
    status: Literal["success"] = "success"
    code: int = 200
    message: str = "ok"
    data: dict[str, Any]
    meta: dict[str, Any] = {}
