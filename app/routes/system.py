"""GET /health · /ready · /info — fuera del lock GPU 🔓."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict:
    return {
        "status": "success",
        "code": 200,
        "message": "up",
        "data": {"service": "up"},
        "meta": {},
    }


@router.get("/ready")
async def ready(request: Request):
    svc = request.app.state.svc
    if svc.ready:
        return {
            "status": "success",
            "code": 200,
            "message": "ready",
            "data": {
                "ready": True,
                "model_loaded": True,
                "device": svc.device,
                "attn": svc.attn_used,
            },
            "meta": {"model": svc.s.model_name},
        }
    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "code": 503,
            "message": "model not loaded yet",
            "error": {
                "type": "not_ready",
                "details": [
                    {
                        "field": "model",
                        "reason": "model_not_loaded",
                        "message": "modelo aún no cargado, reintenta en un momento",
                    }
                ],
            },
            "meta": {"model": svc.s.model_name},
        },
    )


@router.get("/info")
async def info(request: Request) -> dict:
    """Descriptor para agentes 🤖: límites, dims, ejemplos curl."""
    svc = request.app.state.svc
    s = svc.s
    base = f"http://{s.host}:{s.port}"
    return {
        "status": "success",
        "code": 200,
        "message": "voyage-4-nano embedding api",
        "data": {
            "model": s.model_name,
            "endpoints": ["POST /embed/query", "POST /embed/document"],
            "limits": {
                "max_texts": s.max_texts,
                "max_chars": s.max_chars,
                "max_tokens": s.max_tokens,
            },
            "truncate_dims": [2048, 1024, 512, 256],
            "precisions": ["float32", "binary", "ubinary"],
            "notes": [
                "texts siempre es lista, incluso para 1 texto 📝",
                "truncate_dim es requerido: a menor dim, menos detalle ⚠️",
                "embeddings[i] corresponde a texts[i], orden garantizado 🔗",
                "fail-fast: si 1 texto falla, falla todo el request (422) 🔒",
            ],
            "examples": {
                "curl_query": (
                    f"curl -X POST {base}/embed/query "
                    "-H 'Content-Type: application/json' -d "
                    '\'{"texts": ["Which planet is the Red Planet?"], '
                    '"truncate_dim": 512}\''
                ),
                "curl_document": (
                    f"curl -X POST {base}/embed/document "
                    "-H 'Content-Type: application/json' -d "
                    '\'{"texts": ["Mars is known as the Red Planet."], '
                    '"truncate_dim": 512}\''
                ),
            },
        },
        "meta": {"model": s.model_name},
    }
