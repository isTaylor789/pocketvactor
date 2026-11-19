# SPEC — `POST /embed/query` 🔍

> Embeddings de queries (preguntas / búsquedas). Usa el prompt `query` del modelo:
> `"Represent the query for retrieving supporting documents: "`.
> Sobre común: ver `SPEC-errors.md`. Límites globales: ver `SPEC-GLOBAL.md`.

## 1. Request 📥

Siempre lista, incluso para 1 texto (lista de 1). Sin endpoint single separado. 🧬

```jsonc
{
  "texts": ["Which planet is known as the Red Planet?"],  // list[str], 1..64, cada uno 1..32000 chars
  "truncate_dim": 512,    // requerido — ver §3. Sin default mágico.
  "precision": "float32", // opcional, default "float32"
  "normalize": true       // opcional, default true
}
```

Equivalencia Pydantic (referencia):

```python
class EmbedQueryRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    truncate_dim: Literal[2048, 1024, 512, 256]  # requerido
    precision: Literal["float32","binary","ubinary"] = "float32"  # int8/uint8 sin soporte upstream
    normalize: bool = True

    @field_validator("texts")
    @classmethod
    def _check_texts(cls, v):
        for i, t in enumerate(v):
            if not isinstance(t, str): raise ValueError(f"texts[{i}]: not_string")
            if t == "": raise ValueError(f"texts[{i}]: empty")
            if t.strip() == "": raise ValueError(f"texts[{i}]: whitespace_only")
            if len(t) > 32000: raise ValueError(f"texts[{i}]: exceeds_max_chars ({len(t)}>32000)")
        return v
```

Procesamiento interno: chunking de 32 (`64 textos = 32+32` pasadas bajo `Lock(1)`), tokenización con `truncation=True, max_length=32768`, prompt de query prependeado automáticamente, `truncated[i]=true` si se cortó. ✂️

## 2. Response 📤

`200` + sobre `success` (`SPEC-errors.md` §2.1). `embeddings[i] ↔ texts[i]`, orden garantizado.

```jsonc
{
  "status": "success", "code": 200, "message": "embedded 1 texts",
  "data": { "embeddings": [[0.01, 0.02]], "dim": 512, "precision": "float32", "count": 1 },
  "meta": { "model": "voyage-4-nano", "endpoint": "query", "tokens": 24, "truncated": [false], "normalize": true, "ms": 18 }
}
```

curl:

```bash
curl -X POST localhost:8000/embed/query -H 'Content-Type: application/json' -d '{
  "texts": ["Which planet is known as the Red Planet?"],
  "truncate_dim": 512
}'
```

## 3. `truncate_dim` — por qué es requerido ⚠️

La dimensión implica pérdida de detalle (MRL). Si el servidor asumiera un default silencioso, el agente no sabría que está perdiendo calidad. Por eso:

- Es **requerido** en el body (o query param — a fijar en implementación, pero requerido igual).
- Cualquier valor fuera de `{2048,1024,512,256}` → `422 invalid_param/invalid_truncate_dim` con `allowed` ecoado.
- Guía para el agente (también en `GET /info`): textos largos con detalle → `2048`; frases cortas / clasificación → `512` o `256`. 💡

## 4. Edge cases del input 😈 (validación obligatoria)

| # | Caso | Input ejemplo | Esperado |
|---|---|---|---|
| 1 | Lista vacía | `{"texts": [], ...}` | `422 validation_error/empty_list` — "manda al menos 1 texto" |
| 2 | Falta `texts` | `{}` | `422` — `field: texts, reason: missing` |
| 3 | `texts` no es lista | `{"texts": "hola"}` | `422` — `reason: not_a_list`. Ojo agente: aunque sea 1 texto, va en lista. 📝 |
| 4 | Demasiados textos | 65 elementos | `422 too_many_texts` — `received: 65, max: 64` |
| 5 | Item `null` | `{"texts": ["ok", null]}` | `422 null_item` en `texts[1]` |
| 6 | Item no-string | `{"texts": ["ok", 123, true]}` | `422 not_string` en `texts[1]`, con `received_type: number` |
| 7 | String vacío | `{"texts": [""]}` | `422 empty` en `texts[0]` |
| 8 | Solo espacios/tabs/newlines | `{"texts": ["   \n\t  "]}` | `422 whitespace_only` — el modelo devolvería basura, se rechaza. 🗑️ |
| 9 | Excede 32000 chars | texto de 32001 | `422 exceeds_max_chars` con `received_chars` y `max_chars`. Mensaje: "parte el texto o resume". |
| 10 | Límite exacto 32000 | 32000 chars | ✅ pasa (límite inclusivo). Test de borde. |
| 11 | Tokens > 32768 (CJK / código denso / repetido) | ~32000 chars pero >32k tokens | ✅ pasa, se trunca, `meta.truncated[i]=true`. No es error, es warning. ✂️ |
| 12 | Anidado | `{"texts": [["hola"]]}` | `422 not_string` (no se aplana en silencio). |
| 13 | Objeto en vez de string | `{"texts": [{"text": "hola"}]}` | `422 not_string` — "usa string plano, no {text:}". |
| 14 | `truncate_dim` inválido | `100`, `"512"`, `512.0`, `null` | `422 invalid_param/invalid_truncate_dim` con `allowed: [2048,1024,512,256]`. `"512"` string → error (no coerción silenciosa). 🔒 |
| 15 | Falta `truncate_dim` | sin campo | `422` — `reason: missing`, "es requerido, elige según detalle". |
| 16 | `precision` inválida | `"int4"`, `"FLOAT32"`, `"int8"`, `"uint8"` | `422 invalid_precision`. Case-sensitive a propósito (los agentes inventan mayúsculas). `int8`/`uint8` no tienen soporte upstream. |
| 17 | `normalize` no-bool | `"true"`, `1`, `null` | `422 invalid_normalize`. No coerción (`"false"` string es truthy en JS, trampa clásica 🪤). |
| 18 | Campos extra | `{"texts": [...], "foo": 1}` | ✅ se ignoran (o `422` si se configura strict — a fijar en implementación; recomendado: ignorar + no romper agentes). |
| 19 | Unicode/emoji/RTL | `"Marte 🔴 المريخ 火星"` | ✅ pasa. Contar chars en codepoints Python, no bytes. Nunca cortar a mitad de surrogate. 🌍 |
| 20 | Solo emoji | `"🔴🔴🔴"` | ✅ pasa (no es whitespace). |
| 21 | Texto gigante de 1 char | `"a"*32000` | ✅ pasa (raro pero válido, útil para test de tokenizer). |
| 22 | Duplicados | `["hola","hola"]` | ✅ pasa, se embedden 2 veces (útil para test de determinismo). No deduplicar en silencio. |
| 23 | Orden | `["b","a"]` | ✅ `embeddings[0]` es de `"b"`. Nunca reordenar. 🔗 |
| 24 | Content-Type erróneo | `text/plain` | `422` de FastAPI — mensaje debe decir "usa Content-Type: application/json". |
| 25 | Body no-JSON | `not json{{{` | `422` — "body inválido, manda JSON". |
| 26 | `texts: [123456789]` (número largo, agente confundido) | número | `422 not_string` — "envuélvelo en comillas". |

Fail-fast global: se reportan TODOS los índices malos en `error.details` (no solo el primero), y falla todo el request. 🔒

## 5. Tests mínimos 🧪

- `tests/test_query.py`: happy path 1 texto + N textos (orden), cada edge case 1-17 como parametrizado, `truncate_dim` requerido, `GET /info` lista este endpoint.
