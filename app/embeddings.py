"""Wrapper del modelo con semáforo de 1 GPU, SIN cola. 🔒💥

- Carga única (lifespan) con fallback de atención: flash_attention_2 -> sdpa -> eager.
- 1 request a la vez en GPU via ``asyncio.Lock`` del stdlib (sin librería extra ✅).
- Si llega un 2do request con la GPU ocupada → 503 inmediato, no espera ni encola.
- ``encode_*`` de torch es bloqueante → va a threadpool para no frenar el loop.
"""

import asyncio
import time
from typing import Any

import torch

from .config import Settings


class NotReadyError(Exception):
    """El modelo aún no está cargado (503)."""


class GpuBusyError(Exception):
    """La GPU está ocupada: se rechaza de inmediato (503, sin cola)."""


class GpuTimeoutError(Exception):
    """La inferencia excedió inference_timeout (504)."""


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.model: Any = None
        self.tokenizer: Any = None
        self.lock = asyncio.Lock()  # ⬅️ el semáforo global de 1, stdlib, sin deps
        self.device = "cpu"
        self.attn_used = settings.attn_implementation

    # ---------- carga ----------
    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        use_cuda = torch.cuda.is_available() and self.s.device in ("cuda", "auto")
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.bfloat16 if (use_cuda and self.s.dtype == "bfloat16") else torch.float32

        last_err: Exception | None = None
        for attn in (self.s.attn_implementation, "sdpa", "eager"):
            try:
                self.model = SentenceTransformer(
                    self.s.model_path,
                    trust_remote_code=True,  # obligatorio: modeling_qwen3_bidirectional.py local
                    truncate_dim=2048,
                    device=self.device,
                    model_kwargs={"dtype": dtype, "attn_implementation": attn},
                )
                self.attn_used = attn
                break
            except Exception as e:  # noqa: BLE001 — probamos el siguiente fallback
                last_err = e
        if self.model is None:
            raise RuntimeError(f"no se pudo cargar el modelo: {last_err}")
        self.tokenizer = self.model.tokenizer

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ---------- inferencia ----------
    _QUERY_PROMPT = "Represent the query for retrieving supporting documents: "
    _DOC_PROMPT = "Represent the document for retrieval: "

    def _token_stats(self, texts: list[str], kind: str) -> tuple[int, list[bool]]:
        """Tokens totales (con prompt incluido) + flag de truncado por texto."""
        prefix = self._QUERY_PROMPT if kind == "query" else self._DOC_PROMPT
        # len del prompt una sola vez (mismo para todos los textos del request)
        try:
            prompt_n = len(self.tokenizer.encode(prefix, add_special_tokens=False))
        except Exception:
            prompt_n = 0
        total = 0
        truncated = []
        for t in texts:
            try:
                n = len(self.tokenizer.encode(prefix + t, add_special_tokens=True))
            except Exception:
                n = len(self.tokenizer.encode(t, add_special_tokens=True)) + prompt_n
            total += min(n, self.s.max_tokens)
            truncated.append(n > self.s.max_tokens)
        return total, truncated

    def _encode_sync(
        self, kind: str, texts: list[str], truncate_dim: int, precision: str, normalize: bool
    ):
        fn = self.model.encode_query if kind == "query" else self.model.encode_document
        # 1) Siempre pedimos float32 a ST: su path cuantizado hace
        #    Tensor.cpu().numpy() y revienta con bf16 ("Got unsupported
        #    ScalarType BFloat16"). El path float32 sí hace .float().numpy().
        # 2) Cuantizamos nosotros sobre numpy float32 (ya normalizado/truncado).
        vecs = fn(
            texts,
            batch_size=self.s.internal_batch_size,  # 64 textos = 32+32 pasadas 🔁
            precision="float32",
            normalize_embeddings=normalize,
            truncate_dim=truncate_dim,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if precision != "float32":
            from sentence_transformers.util.quantization import quantize_embeddings

            vecs = quantize_embeddings(vecs, precision=precision)  # type: ignore[arg-type]
        return vecs

    async def embed(
        self, kind: str, texts: list[str], truncate_dim: int, precision: str, normalize: bool
    ) -> dict:
        if not self.ready:
            raise NotReadyError("model_not_loaded")
        # Semáforo 1 SIN cola: si la GPU está ocupada se rechaza al momento
        # (503 gpu_busy). Nada de FIFO ni esperar: el 2do request a la mierda. 💥
        if self.lock.locked():
            raise GpuBusyError("lock_timeout")
        await self.lock.acquire()
        try:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            try:
                vecs = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._encode_sync(kind, texts, truncate_dim, precision, normalize),
                    ),
                    timeout=self.s.inference_timeout,
                )
            except TimeoutError:
                raise GpuTimeoutError("inference_timeout") from None
            ms = int((time.perf_counter() - t0) * 1000)
        finally:
            self.lock.release()

        tokens, truncated = self._token_stats(texts, kind)
        return {
            "embeddings": vecs.tolist(),
            "dim": truncate_dim,
            "precision": precision,
            "count": len(texts),
            "meta": {
                "model": self.s.model_name,
                "endpoint": kind,
                "tokens": tokens,
                "truncated": truncated,
                "normalize": normalize,
                "ms": ms,
            },
        }
