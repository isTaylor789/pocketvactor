"""POST /embed/query 🔍 — prompt `query` aplicado por el modelo."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..schemas import EmbedRequest
from ._shared import handle_embed

router = APIRouter(tags=["embed"])


@router.post("/embed/query")
async def embed_query(body: EmbedRequest, request: Request) -> JSONResponse:
    return await handle_embed("query", body, request)
