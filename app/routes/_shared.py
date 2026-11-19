"""Lógica compartida de /embed/* — valida → lock GPU → responde sobre success. 🔗"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from ..embeddings import GpuBusyError, GpuTimeoutError, NotReadyError
from ..schemas import EmbedRequest, ErrorBody, ErrorEnvelope, SuccessEnvelope

log = logging.getLogger("voyage.embed")


def _err(
    code: int, message: str, typ: str, details: list[dict], kind: str, model: str
) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=ErrorEnvelope(
            code=code,
            message=message,
            error=ErrorBody(type=typ, details=details),  # type: ignore[arg-type]
            meta={"model": model, "endpoint": kind},
        ).model_dump(),
    )


async def handle_embed(kind: str, body: EmbedRequest, request: Request) -> JSONResponse:
    svc = request.app.state.svc
    try:
        result = await svc.embed(
            kind, body.texts, body.truncate_dim, body.precision, body.normalize
        )
    except NotReadyError:
        log.info("embed_error endpoint=%s type=not_ready", kind)
        return _err(
            503,
            "model not loaded yet, retry /ready",
            "not_ready",
            [
                {
                    "field": "model",
                    "reason": "model_not_loaded",
                    "message": "modelo aún no cargado, reintenta /ready",
                }
            ],
            kind,
            svc.s.model_name,
        )
    except GpuBusyError:
        log.info("embed_error endpoint=%s type=gpu_busy", kind)
        return _err(
            503,
            "gpu busy, retry in a moment",
            "gpu_busy",
            [
                {
                    "field": "gpu",
                    "reason": "lock_timeout",
                    "message": "GPU ocupada por otro request, reintenta en un momento",
                }
            ],
            kind,
            svc.s.model_name,
        )
    except GpuTimeoutError:
        log.info("embed_error endpoint=%s type=gpu_timeout", kind)
        return _err(
            504,
            "inference exceeded timeout, try fewer/shorter texts",
            "gpu_timeout",
            [
                {
                    "field": "texts",
                    "reason": "inference_timeout",
                    "message": "inferencia excedió 60s, prueba con menos textos o más cortos",
                }
            ],
            kind,
            svc.s.model_name,
        )

    meta = result.pop("meta")
    # Log estructurado sin vectores: debug sin leer bodies. 🔒
    trunc_n = sum(1 for t in meta.get("truncated", []) if t)
    log.info(
        "embed_ok endpoint=%s count=%d dim=%s precision=%s tokens=%d ms=%d truncated=%d",
        kind,
        result["count"],
        result["dim"],
        result["precision"],
        meta.get("tokens"),
        meta.get("ms"),
        trunc_n,
    )
    return JSONResponse(
        status_code=200,
        content=SuccessEnvelope(
            code=200,
            message=f"embedded {result['count']} texts",
            data=result,
            meta=meta,
        ).model_dump(),
    )
