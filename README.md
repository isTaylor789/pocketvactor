# PocketVector 

<p align="center">
  <img src="img/64d9a4ae.png" alt="PocketVector — Local AI Embeddings API" width="100%">
</p>

<p align="center">
  <strong>Fast, predictable embeddings. Running locally.</strong><br>
  API HTTP local de embeddings sobre <code>voyage-4-nano</code>.
</p>

**PocketVector** pone `voyage-4-nano` en tu bolsillo: una API local, rápida y
predecible para generar embeddings de **query** y **document** sin salir de tu
máquina.

> Pensada para agentes y scripts locales que necesitan vectores consistentes,
> errores claros y cero dependencia de servicios externos.

- **Query + document** — prompts separados para retrieval real.
- **GPU local** — un request a la vez, sin colas ni sorpresas.
- **Contrato estable** — sobre único, códigos HTTP reales y errores por índice.
- **Testeable sin GPU** — validación, edge cases y concurrencia simulada.

No es un servicio público ni un servidor de alto throughput: es una API local,
con validaciones estrictas y límites explícitos porque la consumen agentes.

---

## 1. ¿Qué es PocketVector? 🧭

Esta API expone el modelo `voyage-4-nano` por HTTP para dos usos:

- **`/embed/query`** — embeddings de preguntas/búsquedas.
- **`/embed/document`** — embeddings de pasajes/documentos para indexar.

Los dos endpoints usan **prompts distintos** (`query` y `document`). Por eso
**no se deben mezclar** vectores de un endpoint con el otro: pertenecen a
espacios de retrieval diferentes.

La API devuelve los embeddings en el body, con un sobre uniforme
`{status, code, message, data|error, meta}`. El detalle de cada error indica
**qué campo, qué índice y por qué** falló.

### Capacidades ✨

- Listas de 1 a 64 textos por request.
- `truncate_dim` requerido: `2048 | 1024 | 512 | 256` (MRL).
- `precision`: `float32 | binary | ubinary`.
- `normalize`: `true` por defecto.
- Batching interno en GPU (`internal_batch_size=32`).
- Un solo request usa la GPU a la vez, **sin cola**.
- Rate-limit local suave para `/embed/*`.
- `/health`, `/ready`, `/info` y `/docs` fuera del lock de GPU.

### Qué **no** soporta 🚫

- `int8` y `uint8`: no tienen soporte upstream en `voyage-4-nano`; se rechazan
  con `422 invalid_precision`.
- Mezclar `query` y `document` en un mismo request: son endpoints separados.
- Autenticación, multi-GPU, vLLM, vector DB o persistencia.
- `internal_batch_size` configurable por request.

---

## 2. Instalación 🧰

### Requisitos

- Python 3.11+ (el proyecto está probado con 3.12/3.14 en el entorno local).
- GPU recomendada: RTX 3060 12 GB o similar.
- `torch` con CUDA instalado según la versión de tu driver.
- El modelo debe estar en `./model` (o ajustar `VOYAGE_MODEL_PATH`).

### Crear entorno e instalar

```bash
python -m venv env
source env/bin/activate

pip install -r requirements.txt
```

> Si tu entorno se llama distinto, usa su intérprete. Los scripts de este repo
> asumen `env/bin/python` (con fallback a `.venv/bin/python`).

### Modelo: no viene incluido ⚠️

La carpeta `model/` **no se versiona** y **no se descarga automáticamente**. Tienes que
conseguir los archivos del modelo oficial y dejarlos en `./model` (o cambiar
`VOYAGE_MODEL_PATH`).

> 🚫 **No uses `huggingface-cli download` ni `snapshot_download`.** Esas
> herramientas pueden crear una estructura de carpetas/symlinks distinta y
> romper las rutas que espera `SentenceTransformer` (sobre todo
> `1_Pooling/config.json`). Descarga **archivo por archivo** respetando la
> estructura.

Estructura mínima requerida:

```text
model/
├── 1_Pooling/
│   └── config.json
├── config.json
├── config_sentence_transformers.json
├── merges.txt
├── model.safetensors
├── modeling_qwen3_bidirectional.py
├── modules.json
├── sentence_bert_config.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

Archivos opcionales (no necesarios para inferencia):

```text
model/LICENSE.txt
model/NOTICE.txt
model/README.md
model/gitattributes
model/docs/MongoDB_Spring-Green.svg
```

Puntos críticos:

- `1_Pooling/config.json` debe quedar **dentro de `1_Pooling/`**, no en la raíz.
- `modeling_qwen3_bidirectional.py` debe quedar en la raíz de `model/`;
  es obligatorio porque el modelo se carga con `trust_remote_code=True`.
- `model.safetensors` pesa ~692 MB; el modelo completo ronda los **672 MB**.
- Si cambias la ruta, ajusta `VOYAGE_MODEL_PATH`.

Ejemplo de descarga archivo por archivo (reemplaza `<org>/<model>` por el repo
oficial):

```bash
BASE="https://huggingface.co/<org>/<model>/resolve/main"

mkdir -p model/1_Pooling
for f in \
  config.json \
  config_sentence_transformers.json \
  merges.txt \
  model.safetensors \
  modeling_qwen3_bidirectional.py \
  modules.json \
  sentence_bert_config.json \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json \
  1_Pooling/config.json
do
  curl -L --create-dirs "$BASE/$f" -o "model/$f"
done
```

Verifica que quedó bien antes de arrancar:

```bash
ls model/model.safetensors model/modeling_qwen3_bidirectional.py model/1_Pooling/config.json
```

### Configuración

```bash
cp .env.example .env
```

Variables principales (todas con prefijo `VOYAGE_`):

| Variable | Default | Qué hace |
|---|---|---|
| `VOYAGE_MODEL_PATH` | `./model` | Ruta del modelo local. |
| `VOYAGE_HOST` | `127.0.0.1` | Host del server. |
| `VOYAGE_PORT` | `9009` | Puerto del server. |
| `VOYAGE_DEVICE` | `cuda` | `cuda`, `cpu` o `auto`. |
| `VOYAGE_DTYPE` | `bfloat16` | Dtype de inferencia. |
| `VOYAGE_ATTN_IMPLEMENTATION` | `flash_attention_2` | Fallback automático a `sdpa`/`eager`. |
| `VOYAGE_MAX_TEXTS` | `64` | Máximo de textos por request. |
| `VOYAGE_MAX_CHARS` | `32000` | Máximo de chars por texto. |
| `VOYAGE_MAX_TOKENS` | `32768` | Máximo de tokens por texto; se trunca y se avisa. |
| `VOYAGE_INTERNAL_BATCH_SIZE` | `32` | Sub-batches internos en GPU. |
| `VOYAGE_INFERENCE_TIMEOUT` | `60` | Timeout de inferencia en segundos. |
| `VOYAGE_RATE_LIMIT_PER_MINUTE` | `120` | Rate-limit local de `/embed/*`. |
| `VOYAGE_SKIP_MODEL_LOAD` | `false` | `1` = modo validación sin GPU (tests). |

`VOYAGE_LOCK_TIMEOUT` queda **reservado y sin efecto**: no hay cola; si la GPU
está ocupada, el segundo request se rechaza al momento.

---

## 3. Arrancar el server 🚀

```bash
env/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 9009
```

Verificación:

```bash
curl -s localhost:9009/health
curl -s localhost:9009/ready
```

`/ready` devuelve `200` cuando el modelo ya está cargado y `503` mientras no lo
esté. La carga del modelo ocurre **una sola vez** en el `lifespan` de FastAPI.

Swagger: `http://127.0.0.1:9009/docs`
OpenAPI: `http://127.0.0.1:9009/openapi.json`

---

## 4. Endpoints 🛰️

### 4.1 `POST /embed/query` 🔍

Embeddings de preguntas/búsquedas. Aplica el prompt:

```text
Represent the query for retrieving supporting documents:
```

Request:

```json
{
  "texts": ["Which planet is known as the Red Planet?"],
  "truncate_dim": 512,
  "precision": "float32",
  "normalize": true
}
```

Reglas:

- `texts` siempre es **lista**, incluso para 1 texto.
- `truncate_dim` es **requerido**; no hay default oculto.
- `precision` default `float32`.
- `normalize` default `true`.
- `embeddings[i]` corresponde a `texts[i]`; el orden se garantiza.
- Fail-fast: si un texto falla validación, falla **todo** el request.

```bash
curl -X POST localhost:9009/embed/query \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": ["Which planet is known as the Red Planet?"],
    "truncate_dim": 512
  }'
```

### 4.2 `POST /embed/document` 📄

Embeddings de documentos/pasajes para indexar. Mismo schema que `/embed/query`,
pero aplica el prompt:

```text
Represent the document for retrieval:
```

```bash
curl -X POST localhost:9009/embed/document \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": ["Mars, known for its reddish appearance, is often referred to as the Red Planet."],
    "truncate_dim": 512
  }'
```

> El mismo texto en `/embed/query` y `/embed/document` produce **vectores
> distintos** porque los prompts son distintos. Es intencional.

### 4.3 `GET /health`

Liveness. No toca el modelo ni el lock de GPU.

```json
{
  "status": "success",
  "code": 200,
  "message": "up",
  "data": { "service": "up" },
  "meta": {}
}
```

### 4.4 `GET /ready`

Readiness. `200` si el modelo está cargado; `503 not_ready` si todavía no.

```json
{
  "status": "success",
  "code": 200,
  "message": "ready",
  "data": {
    "ready": true,
    "model_loaded": true,
    "device": "cuda",
    "attn": "sdpa"
  },
  "meta": { "model": "voyage-4-nano" }
}
```

`attn` muestra la implementación de atención realmente usada; si
`flash_attention_2` falla, se usa `sdpa` o `eager`.

### 4.5 `GET /info`

Descriptor para agentes: límites, dims, precisions, notas y ejemplos `curl`.

### 4.6 `GET /docs` y `GET /openapi.json`

Documentación interactiva y contrato OpenAPI generados por FastAPI.

---

## 5. Respuesta estándar 📦

### Success

```json
{
  "status": "success",
  "code": 200,
  "message": "embedded 2 texts",
  "data": {
    "embeddings": [[0.01, 0.02], [0.03, 0.04]],
    "dim": 512,
    "precision": "float32",
    "count": 2
  },
  "meta": {
    "model": "voyage-4-nano",
    "endpoint": "query",
    "tokens": 24,
    "truncated": [false, false],
    "normalize": true,
    "ms": 42
  }
}
```

### Error

```json
{
  "status": "error",
  "code": 422,
  "message": "validation failed (1): texts[1]: está vacío...",
  "error": {
    "type": "validation_error",
    "details": [
      {
        "field": "texts[1]",
        "reason": "empty",
        "message": "texts[1] está vacío, manda al menos 1 caracter",
        "index": 1
      }
    ]
  },
  "meta": { "model": "voyage-4-nano", "endpoint": "query" }
}
```

Reglas del sobre:

- `code` **siempre** coincide con el HTTP status real.
- `status` es `success` o `error`.
- `data` y `error` son excluyentes.
- Cada `detail` trae `{field, reason, message}`; `index` aparece cuando aplica.
- Los embeddings **nunca** se escriben en el log del server.

### Códigos y `error.type`

| HTTP | `error.type` | Cuándo |
|---:|---|---|
| 422 | `validation_error` | Input malformado: `empty_list`, `missing`, `not_a_list`, `too_many_texts`, `empty`, `whitespace_only`, `not_string`, `null_item`, `exceeds_max_chars`, `invalid_json`, `invalid_body`. |
| 422 | `invalid_param` | `truncate_dim`, `precision` o `normalize` inválidos. |
| 429 | `rate_limited` | Se superó el rate-limit local de `/embed/*`. |
| 503 | `not_ready` | El modelo todavía no está cargado. |
| 503 | `gpu_busy` | Otro request está usando la GPU; no hay cola. |
| 504 | `gpu_timeout` | La inferencia superó `VOYAGE_INFERENCE_TIMEOUT`. |
| 500 | `internal` | Bug real. Al cliente solo se le da un `request_id`; el detalle queda en el log. |

---

## 6. Concurrencia y rendimiento ⚙️

### Un request a la vez, sin cola

La GPU se protege con un `asyncio.Lock` global. El flujo es:

```text
validar → adquirir lock → encode_* en threadpool → liberar lock → responder
```

- Si la GPU está ocupada, el segundo request recibe **`503 gpu_busy` de
  inmediato**. No espera, no se encola, no hay FIFO.
- `/health`, `/ready` y `/info` quedan fuera del lock.
- `encode_*` es bloqueante (torch), por eso corre en threadpool para no bloquear
  el event loop.

### Batching interno

`internal_batch_size=32` significa que un request de 64 textos se procesa como
`32 + 32` pasadas internas, todo bajo el mismo lock. No es configurable por
request.

### Rate-limit

- Solo aplica a `/embed/*`.
- Default: `120` requests por minuto por IP.
- Al superarlo: `429 rate_limited` con el sobre estándar.
- `/health`, `/ready` e `/info` no consumen cupo.

### Timeout

Si una inferencia supera `VOYAGE_INFERENCE_TIMEOUT` (60 s por defecto), se
devuelve `504 gpu_timeout` y el lock se libera.

### Benchmarks medidos (RTX 3060 12 GB, bf16, `sdpa`, `internal_batch_size=32`)

| Caso | p50 cliente | p95 cliente | `meta.ms` p50 | textos/s |
|---|---:|---:|---:|---:|
| 1 texto, dim 512 | 39.4 ms | 40.6 ms | 34 | 25.3 |
| 8 textos, dim 512 | 45.1 ms | 48.4 ms | 33 | 175.8 |
| 32 textos, dim 512 | 78.2 ms | 81.0 ms | 41 | 405.3 |
| 64 textos, dim 512 | 158.3 ms | 161.1 ms | 94 | 404.2 |
| 8 textos, dim 2048 | 72.5 ms | 84.6 ms | 40 | 107.3 |

Comparación de `internal_batch_size` (modelo directo):

| batch | p50 | textos/s |
|---:|---:|---:|
| 16 | 147.9 ms | 432.7 |
| 32 | 127.5 ms | 501.9 |
| 64 | 115.7 ms | 553.0 |

VRAM medida con `nvidia-smi` (RTX 3060 12 GB, bf16, `sdpa`, `internal_batch_size=32`):

| Estado | VRAM del proceso server |
|---|---:|
| Idle (modelo cargado, sin requests) | **784 MiB** |
| Tras inferencias (1–64 textos) | **~846 MiB** |

El proceso incluye **~660 MiB de pesos en bf16** más contexto CUDA y workspaces
de kernels. La VRAM **no crece con las requests**: se estanca en ~846 MiB porque
el caching allocator de PyTorch retiene la memoria; no es una fuga. El total de
la GPU ronda **~1.43–1.49 GB** incluyendo escritorio y overhead del driver.

> ⚠️ Estas cifras son con pesos **bf16**. Con `float32` los pesos solos ocupan
> ~1.3 GB, y con otras cuantizaciones del modelo la VRAM cambia. Los números
> dependen del driver, la versión de torch y la carga de la máquina.
>
> Comparativa GPU vs CPU y más detalles: [`docs/PERF-gpu-vs-cpu.md`](docs/PERF-gpu-vs-cpu.md).

---

## 7. Límites y validación 🛡️

| Límite | Valor |
|---|---:|
| Textos por request | 64 |
| Chars por texto | 32000 |
| Tokens por texto | 32768 |
| Dims permitidas | `2048, 1024, 512, 256` |
| Precisions permitidas | `float32, binary, ubinary` |
| Normalize | `true` por defecto |

Notas importantes:

- `max_chars=32000` cuenta **chars**, no bytes. En UTF-8 de 4 bytes puede haber
  hasta 128000 bytes por texto.
- Si un texto supera `max_tokens=32768`, **no es error**: se trunca y
  `meta.truncated[i]` vale `true`.
- `truncate_dim` es estricto: `512.0`, `"512"`, `true` o `null` → `422`.
- `texts` debe ser lista; `"hola"` no es válido, `["hola"]` sí.
- Se rechazan listas vacías, textos vacíos, solo-espacios, `null`, no-string y
  textos que excedan `max_chars`.
- Si un solo texto falla, falla todo el request (`422`), sin parciales.

> ⚠️ Con `max_chars=32000` y hasta 64 textos, un request puede ser muy grande.
> El límite de 60 s y el lock de 1 protegen el server, pero el desarrollador
> decide si conviene bajar `VOYAGE_MAX_CHARS` según su caso de uso.

---

## 8. Calidad numérica y decisiones 🧪

### Precisión soportada

- `float32`: vectores float.
- `binary`: empaquetado `dim/8` bytes con int8.
- `ubinary`: empaquetado `dim/8` bytes con uint8.
- `int8` y `uint8`: **no soportados upstream**; `422 invalid_precision`.

### Normalización

Con `normalize=true`, SentenceTransformers normaliza los vectores. Medido en
esta máquina, la norma queda en **≈1.003** por el ruido de `bfloat16`, no
exactamente 1.0.

- Para **cosine similarity**, esto no afecta el ranking.
- Si se usa **dot product** asumiendo vectores unitarios, el efecto es un factor
  ~1.006 constante; tampoco cambia el ranking.
- Si se necesita `‖v‖=1.0` exacto, se puede renormalizar en float32 al final
  (mejora opcional, no implementada).

### Determinismo y batch

- El mismo request repetido produce **el mismo hash** de vector.
- El mismo texto solo vs. dentro de un batch puede variar ~`0.002` por
  componente debido a bf16.
- Para retrieval con coseno, ese ruido (**0.001–0.004** de coseno) es
  despreciable frente a los gaps típicos relevantes/irrelevantes (0.05–0.5).
- No hay reproducibilidad bit-a-bit entre batches distintos. Si se necesita
  exactitud estricta: usar float32 o codificar cada texto por separado con
  padding fijo, asumiendo el costo.

### Truncado

Con `max_chars=32000` y `max_tokens=32768`, el truncado ahora **sí es
alcanzable** con texto denso (por ejemplo 32000 chars de 4 bytes = 128000
tokens). En ese caso `meta.truncated[i]=true` y `meta.tokens` queda limitado a
32768.

---

## 9. Tests y calidad ✅

```bash
# Tests sin GPU (validación, edge cases, concurrencia simulada, timeout, rate-limit)
./scripts/test.sh
# o:
env/bin/python -m pytest tests/ -q

# Lint + formato
env/bin/ruff check app/ tests/
env/bin/ruff format --check app/ tests/

# Tipos
env/bin/mypy app/
```

Estado actual:

- **55 tests** pasan sin GPU.
- `ruff` limpio.
- `mypy` limpio.
- `scripts/smoke.sh`: contrato contra server vivo (25 checks).
- Verificación numérica: determinismo, dims, precisions, normas, orden,
  similitud query/document.
- Verificación de contrato: OpenAPI, `/docs`, sobres, `code == HTTP` y catálogo
  de `reason`.

### Scripts

| Script | Qué hace |
|---|---|
| `scripts/quick.sh` | Health, ready y 1 query. |
| `scripts/smoke.sh` | Suite de contrato contra server vivo: precisions, dims, 422s, concurrencia, D6. |
| `scripts/test.sh` | `pytest` sin GPU; detecta `.venv` o `env`. |
| `scripts/verify/verify_contract.py` | OpenAPI, `/docs`, sobres y catálogo de errores. |
| `scripts/verify/verify_numeric.py` | Determinismo, dims, normas, precisions y prompts. |
| `scripts/verify/verify_concurrency.py` | 1 request procesa, el resto recibe `503 gpu_busy`. |
| `scripts/verify/bench.py` | Latencia y throughput HTTP. |
| `scripts/verify/bench_batch.py` | Comparación de `internal_batch_size`. |

```bash
BASE_URL=http://127.0.0.1:9009 ./scripts/smoke.sh
```

---

## 10. Estructura 🗂️

```text
app/
  main.py               # FastAPI, lifespan, handlers, rate-limit
  config.py             # Settings con prefijo VOYAGE_
  schemas.py            # DTOs + validaciones estrictas
  errors.py             # mapeo Pydantic → {field, reason, message}
  embeddings.py         # modelo, batching, lock GPU, timeout
  routes/
    query.py            # POST /embed/query
    document.py         # POST /embed/document
    system.py           # GET /health, /ready, /info
    _shared.py          # manejo común de /embed/*
docs/
  SPEC-GLOBAL.md
  SPEC-embed-query.md
  SPEC-embed-document.md
  SPEC-errors.md
scripts/
  quick.sh
  smoke.sh
  test.sh
  verify/
    verify_contract.py
    verify_numeric.py
    verify_concurrency.py
    bench.py
    bench_batch.py
tests/
  test_api.py
  test_edges.py
  test_query.py
  test_document.py
  test_resilience.py
```
