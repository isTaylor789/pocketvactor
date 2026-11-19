"""App FastAPI — voyage-4-nano local en :9009 🚀.

- Lifespan: carga el modelo UNA vez (salvo VOYAGE_SKIP_MODEL_LOAD=1 para tests 🧪).
- Handler 422 → sobre {status:error} con details por índice.
- Handler 500 → sobre error con request_id (nunca stacktrace al agente).
- Rate-limit suave local (solo /embed/*): 429 con sobre, /health+/ready+/info libres 🐢.
"""

import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import Settings
from .embeddings import EmbeddingService
from .errors import error_type_for, validation_details
from .routes import document, query, system
from .schemas import EmbedRequest

log = logging.getLogger("voyage.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class SoftRateLimit(BaseHTTPMiddleware):
    """Rate-limit suave en memoria: N req/min por IP, solo /embed/*. Sin Redis, es local. 🐢"""

    def __init__(self, app: FastAPI, per_minute: int, model_name: str) -> None:
        super().__init__(app)
        self.per_minute = per_minute
        self.model_name = model_name
        self.hits: dict[str, deque[float]] = {}

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if self.per_minute > 0 and request.url.path.startswith("/embed/"):
            ip = request.client.host if request.client else "local"
            now = time.monotonic()
            dq = self.hits.setdefault(ip, deque())
            while dq and now - dq[0] > 60:
                dq.popleft()
            if len(dq) >= self.per_minute:
                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "error",
                        "code": 429,
                        "message": f"rate limit: {self.per_minute}/min, espera y reintenta",
                        "error": {
                            "type": "rate_limited",
                            "details": [
                                {
                                    "field": "-",
                                    "reason": "rate_limited",
                                    "message": "límite local excedido, reintenta en un momento",
                                }
                            ],
                        },
                        "meta": {"model": self.model_name},
                    },
                )
            dq.append(now)
        return await call_next(request)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    EmbedRequest.configure(settings.max_texts, settings.max_chars)
    svc = EmbeddingService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = svc
        app.state.settings = settings
        if not settings.skip_model_load:
            svc.load()  # ~660MB a VRAM, una sola vez ⏳
        yield

    app = FastAPI(title="voyage-4-nano local", version="0.1.0", lifespan=lifespan)
    app.state.svc = svc
    app.state.settings = settings
    app.add_middleware(
        SoftRateLimit,  # type: ignore[arg-type]
        per_minute=settings.rate_limit_per_minute,
        model_name=settings.model_name,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = validation_details(exc)
        # endpoint legible para el agente: query|document|system
        path = request.url.path
        endpoint = (
            "query" if "/query" in path else ("document" if "/document" in path else "system")
        )
        first = details[0].get("message", "invalid input") if details else "invalid input"
        log.info("embed_rejected endpoint=%s issues=%d first=%s", endpoint, len(details), first)
        body = {
            "status": "error",
            "code": 422,
            "message": f"validation failed ({len(details)}): {first}",
            "error": {"type": error_type_for(details), "details": details},
            "meta": {"model": settings.model_name, "endpoint": endpoint},
        }
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        rid = uuid.uuid4().hex[:8]
        print(f"[error {rid}] {type(exc).__name__}: {exc}")  # log servidor, no al agente
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "code": 500,
                "message": f"internal error (ref {rid})",
                "error": {
                    "type": "internal",
                    "details": [
                        {
                            "field": "-",
                            "reason": "internal_error",
                            "message": f"error interno (ref {rid}), revisa el log del servidor",
                            "request_id": rid,
                        }
                    ],
                },
                "meta": {"model": settings.model_name},
            },
        )

    app.include_router(system.router)
    app.include_router(query.router)
    app.include_router(document.router)
    return app


app = create_app()
