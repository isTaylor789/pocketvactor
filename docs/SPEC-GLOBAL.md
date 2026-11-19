# SPEC GLOBAL — API local de embeddings `voyage-4-nano`

> Nivel: requerimientos de alto nivel. No define schemas de request/response
> (eso va en specs por endpoint). Esta es la visión de trabajo acordada en grilling.

## 1. Objetivo

Exponer `voyage-4-nano` (carpeta local `./model`) como API HTTP local para
consumo de agentes (curl / SDK), rápida y eficiente en una máquina con
`RTX 3060 12GB + 31GB RAM + 24 threads CPU`.

No es un servicio público ni un servidor de alto throughput. Es una API
local con validaciones estrictas porque la van a consumir agentes.

## 2. Decisión de inferencia (del README)

El README propone 3 modos, todos sobre PyTorch:

| Modo | Veredicto |
|---|---|
| `Via Transformers` (`AutoModel` + `mean_pool` manual) | Descartado. Misma velocidad, pero pooling/prompts a mano = frágil. |
| `Via Sentence Transformers` (`encode_query` / `encode_document`) | **Elegido.** Batching, prompts (`query`/`document`), `truncate_dim`, `precision` ya resueltos. |
| `Via vLLM` | Descartado para local. Overkill: reserva VRAM, arranque lento, `max_model_len=32768` + `enforce_eager=True` lo hacen más lento en batch pequeño. Solo tendría sentido como servidor de miles de docs/seg. |

Config de carga elegida:

- `SentenceTransformer("./model", trust_remote_code=True)`
- `model_kwargs = { dtype: torch.bfloat16, attn_implementation: "flash_attention_2" }`
- Carga única al arrancar (lifespan), reutilizada por todos los requests.

## 3. Stack definido

- **Lenguaje:** Python 3.11+
- **Web:** FastAPI (Pydantic v2) + Uvicorn
- **ML:** `sentence-transformers`, `transformers`, `torch` (CUDA 13.x), `huggingface-hub` si hace falta
- **Config:** `pydantic-settings` (env: `MODEL_PATH`, `DEVICE`, `BATCH_SIZE`, etc.)
- **Tests:** `pytest` + `httpx` (`TestClient` / `ASGITransport`)
- **Calidad:** `ruff` (lint+format). `mypy` opcional fase 2.
- **Docs de API:** `/docs` (Swagger) + `/health`, `/ready`
- **Sin Docker fase 1.** Fase 2 opcional.

Estructura esperada (no creada aún):

```
app/
  main.py        # FastAPI + lifespan (carga modelo una vez)
  config.py      # settings globales
  schemas.py     # bases compartidas (sin schemas finales de endpoints)
  embeddings.py  # wrapper del modelo + GPU lock
  routes/
    query.py
    document.py
  errors.py      # formato de error para agentes
docs/
  SPEC-GLOBAL.md        # este archivo
  SPEC-embed-query.md   # pendiente
  SPEC-embed-document.md# pendiente
  SPEC-errors.md        # pendiente (success vs error para curl/agentes)
```

## 4. Endpoints (alto nivel, sin schemas)

- `POST /embed/query` — embeddings de queries (prompt `query`).
- `POST /embed/document` — embeddings de documentos (prompt `document`).
- Separados a propósito: más fácil de mantener y con personalidad propia,
  aunque compartan parámetros de configuración.
- `GET /health` — liveness (proceso vivo).
- `GET /ready` — readiness (modelo cargado en GPU/CPU y listo).
- Futuras specs (`SPEC-embed-query.md`, `SPEC-embed-document.md`) definirán
  request/response exactos. Aquí solo se fija el contrato común.

Contrato común acordado (elementos, no schema final):

- Input: `texts: list[str]` (solo lista, incluso para 1 texto = lista de 1).
- Params modulares por request: `truncate_dim`, `precision`, `normalize`
  (ver §6).
- Output contiene: `embeddings`, `dim`, `precision`, `count`, `model`,
  `usage.tokens`. Schema exacto pendiente.

## 5. Validación y límites (filosofía: API hostil)

Aunque sea local, la consumen agentes → validar todo, fallar con `4xx`
explicando **qué índice falló y por qué**, nunca `500` genérico ni colgar la GPU.

Límites acordados (claridad, no modo rudo):

- `max_texts_per_request = 64`
- `max_chars_per_text = 32000`
- Rechazar: lista vacía, textos vacíos / solo-espacios, `null`, no-string.
- Tokenizer: `truncation=True, max_length=32768`, con aviso cuando se trunca
  (campo de warning en respuesta, formato en SPEC-errors).
- Error por índice: p.ej. `texts[7]: vacío`, `texts[12]: excede 32000 chars`.
  Si un índice falla, falla todo el request (`422`), no retornos parciales
  silenciosos (decisión a confirmar en SPEC-errors si se quiere modo parcial).

## 6. Parámetros de embedding (modular, importante)

Este punto es crítico: la dimensión implica pérdida de detalle, así que el
agente/usuario elige según el caso (textos grandes → dim grande, frases
cortas para clasificar → dim chica).

- `truncate_dim ∈ {2048, 1024, 512, 256}` (MRL). **Sin default mágico
  oculto en la spec final**: o es requerido o el default queda explícito y
  visible en la respuesta (`dim` efectivo). Cualquier otro valor → `422`.
  Recomendación inicial de default si se exige uno: `512`.
- `precision ∈ {float32, binary, ubinary}`. `int8`/`uint8` **no** tienen soporte upstream en voyage-4-nano: se rechazan con `422`.
  Default inicial propuesto: `float32`. Validar, otro valor → `422`.
- `normalize: bool`. Default inicial propuesto: `true`.
- Los defaults son **compartidos** entre `/query` y `/document`, pero cada
  endpoint puede sobreescribirlos por request (query params o body — a definir
  en specs por endpoint).
- La respuesta siempre debe ecoar `dim` y `precision` efectivos para que el
  agente sepa qué recibió.

## 7. Concurrencia y rendimiento (semaforización)

Tres números distintos, no confundir:

| Concepto | Valor | Qué significa |
|---|---|---|
| `max_texts_per_request` | 64 | Límite de API por JSON. |
| `internal_batch_size` | 32 | Sub-batches con los que la GPU procesa un request (64 textos = 32+32 pasadas). Fijo interno fase 1, no configurable por request. |
| `gpu_concurrency` | 1 | Un solo request usa la GPU a la vez. |

Implementación elegida — **manejador de bloqueo, no middleware global**:

- Un `asyncio.Lock()` (o `anyio.Semaphore(1)`) creado en `lifespan`, guardado
  en `app.state`, inyectado por dependencia a ambas rutas.
- ¿Por qué no middleware? El lock solo protege la sección crítica
  (`model.encode_*`), no todo el ciclo HTTP (parseo/validación/logging deben
  seguir en paralelo). Un middleware global serializaría hasta `/health`.
- Flujo: validar → adquirir lock → `await run_in_threadpool(model.encode_*)` →
  liberar → responder. `encode_*` es bloqueante (torch), así que va a
  threadpool para no bloquear el event loop.
- Timeout de ejecución: 60s propuesto → `504` con mensaje
  para agentes (a detallar en SPEC-errors). No hay espera de adquisición:
  si la GPU está ocupada se devuelve `503 gpu_busy` al momento.
- `/health` y `/ready` quedan **fuera** del lock.
- Cola: ninguna. Sin FIFO, sin prioridades, sin pipelines: el 2do request
  concurrente se rechaza con `503` (decisión de proyecto local).

## 8. Observabilidad mínima (para agentes con curl)

- Logs estructurados por request: `endpoint, count, truncate_dim, precision,
  tokens, ms, truncated:bool`. Sin dumpear embeddings al log.
- `usage.tokens` en cada respuesta success para debug sin leer logs.
- Errores: schema estricto y corto (legible por `curl | head`), detalle
  completo en body. Formato exacto en `SPEC-errors.md` (pendiente).

## 9. Plan de trabajo

1. **Fase 0 — Base:** `app/main.py` + `config.py` + `/health` + `/ready` + carga del modelo en lifespan. Criterio: `uvicorn` arranca y `/ready` pasa en la 3060.
2. **Fase 1 — Query:** `SPEC-embed-query.md` → implementar `POST /embed/query` + validaciones §5 + params §6 + lock §7 + tests.
3. **Fase 2 — Document:** `SPEC-embed-document.md` → `POST /embed/document` (reutiliza base, cambia prompt y defaults si aplica) + tests.
4. **Fase 3 — Errores:** `SPEC-errors.md` → unificar success/error, warnings de truncado, códigos (`422` validación, `503/504` GPU ocupada/timeout).
5. **Fase 4 — Endurecer:** rate-limit suave local, tests de concurrencia (2 agentes pegando a la vez), `ruff`, README de uso con curl.

## 10. Fuera de alcance fase 1

- Schemas finales de response (van en specs por endpoint).
- `base64` de embeddings (solo floats directos).
- Autenticación (local sin auth; si se expone, fase 2).
- Multi-proceso / multi-GPU / vLLM.
- `internal_batch_size` configurable por request.
- Persistencia / vector DB / re-index (el espacio compartido voyage-4 permite
  mezclar modelos, pero no se implementa aquí).

## 11. Riesgos conocidos

- `flash_attention_2` requiere build con CUDA; si falla en la máquina, fallback a `sdpa`/`eager` (pequeñas diferencias numéricas esperadas según README).
- Textos cercanos a 32k tokens × batch 32 = OOM. Mitigación: `max_chars=32000` + batch interno 32 + lock 1 + timeout.
- `trust_remote_code=True` es obligatorio (usa `modeling_qwen3_bidirectional.py` local). No desactivar.
