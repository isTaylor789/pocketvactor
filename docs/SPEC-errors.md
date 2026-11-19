# SPEC — Envoltorio común: success / error + utilitarios

> Complementa `SPEC-GLOBAL.md`. Define el sobre que usan TODOS los endpoints
> para que un agente con `curl` sepa si falló sin leerse el log. 🤖

## 1. Sobre estándar 📦

Todos responden HTTP acorde + body con el mismo sobre:

```jsonc
// ✅ success (2xx)
{
  "status": "success",   // literal
  "code": 200,           // repite el HTTP status
  "message": "embedded 2 texts",  // corto, humano
  "data": { ... },       // solo en success
  "meta": { "model": "voyage-4-nano", "endpoint": "query|document|system", "ms": 42 }
}

// ❌ error (4xx/5xx)
{
  "status": "error",
  "code": 422,
  "message": "validation failed: 2 textos inválidos",  // resumen de 1 línea
  "error": { "type": "validation_error", "details": [ ... ] },  // solo en error
  "meta": { "model": "voyage-4-nano", "endpoint": "query|document|system" }
}
```

Reglas:

- `status` ∈ `{"success","error"}`. El agente hace `if status ==` y listo. 🧠
- `code` == HTTP status real. Nunca `200` con error dentro. 🙅
- `message` 1 línea, sin dumps. El detalle va en `error.details`.
- `data` y `error` son excluyentes: success → `data` presente, `error` ausente (y viceversa).
- `meta` siempre presente (aunque sea mínimo) para debug sin logs.
- `embeddings` NUNCA van al log del servidor. Solo al body. 🔒

## 2. `data` en success ✅

### 2.1 `/embed/*`

```jsonc
{
  "embeddings": [[0.01, 0.02, "..."], [0.03, "..."]],  // float[][] — embeddings[i] ↔ texts[i], orden garantizado
  "dim": 512,            // truncate_dim efectivo
  "precision": "float32",// precision efectiva
  "count": 2             // == len(embeddings) == len(texts)
}
```

`meta` en success de embed:

```jsonc
{
  "model": "voyage-4-nano",
  "endpoint": "query",
  "tokens": 128,                 // suma de tokens input (tras prompt), aprox. tokenizer
  "truncated": [false, false],   // truncated[i] por texto (true = se cortó en 32768)
  "normalize": true,
  "ms": 42
}
```

### 2.2 Sistema

- `GET /health` → `200` + `data: { "service": "up" }`. Fuera del lock. 🔓
- `GET /ready` → `200` + `data: { "ready": true, "model_loaded": true, "device": "cuda:0" }`. `503` + sobre error si no listo.
- `GET /info` → `200` (⚠️ no 300: `300 Multiple Choices` es redirect y confunde a curl/agentes) + descriptor para agentes:

```jsonc
{
  "data": {
    "model": "voyage-4-nano",
    "endpoints": ["POST /embed/query", "POST /embed/document"],
    "limits": { "max_texts": 64, "max_chars": 32000, "max_tokens": 32768 },
    "truncate_dims": [2048, 1024, 512, 256],
    "precisions": ["float32", "binary", "ubinary"],
    "examples": { "curl_query": "curl -X POST localhost:8000/embed/query -H 'Content-Type: application/json' -d '...'" }
  }
}
```

## 3. `error` en fallo 💥

```jsonc
{
  "type": "validation_error",  // ver tabla §4
  "details": [                 // 1 entrada por problema, con índice cuando aplique
    { "index": 1, "field": "texts[1]", "reason": "empty", "received": "   " }
  ]
}
```

Campos de cada detail: `field` (siempre), `reason` (código máquina, snake_case), `index` (cuando el fallo es por texto), más contexto (`received_chars`, `max_chars`, `allowed`, `received`). `received` NUNCA incluye el texto completo si es largo: solo preview de 80 chars o conteos. ✂️

## 4. Catálogo de errores 🧾

| HTTP | `error.type` | Cuándo | Ejemplo `reason` |
|---|---|---|---|
| 422 | `validation_error` | Input malformado (ver edge cases en specs de endpoints) | `empty_list`, `too_many_texts`, `empty`, `whitespace_only`, `not_string`, `exceeds_max_chars`, `too_deep_nesting` |
| 422 | `invalid_param` | `truncate_dim`/`precision`/`normalize` fuera de rango | `invalid_truncate_dim`, `invalid_precision`, `invalid_normalize` |
| 503 | `not_ready` | `/ready` o embed antes de cargar modelo | `model_not_loaded` |
| 503 | `gpu_busy` | 2do request con GPU ocupada (sin cola, rechazo inmediato) | `lock_timeout` |
| 504 | `gpu_timeout` | Inferencia excede 60s | `inference_timeout` |
| 429 | `rate_limited` | Rate-limit suave local (>120/min en /embed/*) | `rate_limited` |
| 500 | `internal` | Bug real. Log interno completo, al agente solo id | `internal_error` (+ `request_id`) |

Fail-fast: si 1 texto de 64 falla validación → falla TODO el request con `422` listando todos los índices malos. Sin parciales silenciosos. 🔒

## 5. Códigos `reason` (máquina, para agentes) 🤖

`empty_list | missing | not_a_list | too_many_texts | empty | whitespace_only | not_string | null_item | exceeds_max_chars | invalid_truncate_dim | invalid_precision | invalid_normalize | truncated_input (este es warning en meta, no error) | lock_timeout | inference_timeout | model_not_loaded | internal_error | rate_limited | invalid_json | invalid_body`
