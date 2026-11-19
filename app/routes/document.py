"""POST /embed/document 📄 — prompt `document` aplicado por el modelo."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..schemas import EmbedRequest
from ._shared import handle_embed

router = APIRouter(tags=["embed"])


@router.post("/embed/document")
async def embed_document(body: EmbedRequest, request: Request) -> JSONResponse:
    return await handle_embed("document", body, request)
