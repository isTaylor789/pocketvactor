# SPEC — `POST /embed/document` 📄

> Embeddings de documentos (pasajes / docs para indexar). Usa el prompt `document`:
> `"Represent the document for retrieval: "`.
> Hereda TODO de `SPEC-embed-query.md` salvo lo listado aquí. Sobre común: `SPEC-errors.md`.

## 1. Diferencias vs `/embed/query` 🔀

| Aspecto | `/embed/query` | `/embed/document` |
|---|---|---|
| Prompt interno | `query` | `document` (distinto espacio de retrieval — no mezclar) |
| Uso típico | preguntas cortas | pasajes largos, markdown, código |
| Request schema | idéntico (`texts`, `truncate_dim` requerido, `precision`, `normalize`) | idéntico |
| Response schema | idéntico (sobre `success`) | idéntico, con `meta.endpoint: "document"` |
| Edge cases input | ver query §4 (aplican todos) | aplican todos + §2 de aquí |

Mismo curl, distinto path:

```bash
curl -X POST localhost:8000/embed/document -H 'Content-Type: application/json' -d '{
  "texts": ["Mars, known for its reddish appearance, is often referred to as the Red Planet."],
  "truncate_dim": 512
}'
```

## 2. Edge cases extra de documentos 😈📄

| # | Caso | Esperado |
|---|---|---|
| D1 | Documento markdown largo (`#`, tablas, links) | ✅ pasa, se embedde tal cual. No strippear markdown en silencio. |
| D2 | Código fuente (`def ...`,Llaves) | ✅ pasa. No es "ruido", es contenido válido. |
| D3 | HTML pegado (`<div>...`) | ✅ pasa tal cual fase 1 (sin limpieza). Limpieza HTML = fase 2 si se pide. |
| D4 | Documento que roza 32000 chars pero con muchos tokens (código/CJK) | ✅ pasa, `meta.truncated[i]=true` si se cortó en 32768. |
| D5 | Mezclar query y document en un solo request | 🙅 imposible por diseño (un endpoint = un prompt). Si el agente lo necesita, hace 2 requests. |
| D6 | Mismo texto en query y document | ✅ válido — los vectores SERÁN distintos (prompts distintos). Test útil para probar que no se mezclaron prompts. 🧪 |
| D7 | Doc con solo puntuación (`"...---..."`) | ✅ pasa (no es whitespace). El modelo decide. |
| D8 | Doc con saltos de línea masivos (`\n` x 2000) | ✅ pasa si `strip() != ""`. Normalizar `\r\n` → `\n` en servidor es opcional pero documentarlo. |

## 3. Tests mínimos 🧪

- `tests/test_document.py`: happy path, test D6 (mismo texto ≠ vector query), D4 (truncado con warning), reutilizar parametrizados de query.
