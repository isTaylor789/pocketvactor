# Rendimiento: GPU vs CPU — `voyage-4-nano`

Fecha de medición: **2026-11-19** · PocketVector (API local de embeddings)
Medido sobre el server real (`uvicorn` + FastAPI), 1 request a la vez (respeta el lock de GPU).

## TL;DR

- **La GPU gana en todos los casos**, y la ventaja crece con el tamaño del lote:
  ~**1.9×** con 1 texto, ~**4.5×** con 8 textos, **~10–12×** con 32–64 textos.
- Medido **solo en inferencia** (`meta.ms`), el pico es **~18–21×** con lotes de 32–64.
- La CPU se estanca en **~30–36 textos/s**; la GPU llega a **~371 textos/s**.
- Memoria: GPU **846 MiB VRAM** + **2.05 GB RSS**; CPU **0 VRAM** + **2.58 GB RSS**.
- La latencia cliente incluye serialización (`.tolist()` + JSON) y token stats, por eso el
  speedup de punta a punta es menor que el de inferencia pura.

## Entorno

| | GPU (server :9009) | CPU (server :9010) |
|---|---|---|
| Hardware | NVIDIA GeForce RTX 3060 12 GB (driver 615.78.08) | Intel Xeon E5-2680 v3 @ 2.50 GHz · 24 vCPU |
| `torch` | 2.15.0+cu132 (CUDA 13.2) | 2.15.0+cu132 (CPU) |
| `sentence-transformers` | 6.3.0 | 6.3.0 |
| Python | 3.14.10 | 3.14.10 |
| `dtype` real | `bfloat16` | `float32` |
| `attn` | `sdpa` | `sdpa` |
| Threads torch | — | 12 |
| RAM del sistema | 31 GB | 31 GB |

> `embeddings.py` solo usa `bfloat16` cuando hay CUDA; en CPU cae a `float32`.
> No es una comparación de hardware puro, pero **sí es la configuración real de despliegue**.

## Metodología

- HTTP real contra el server (`POST /embed/query`), textos fijos y cortos (8 pasajes de ~60–90 chars).
- `truncate_dim` y `precision=float32`, `normalize=true` (default).
- 2 requests de warmup + N repeticiones por caso; se reporta **p50/p95 de cliente**.
- `p50 server` = `meta.ms` del server: mide **solo el encode**; no incluye token stats,
  `vec.tolist()` ni serialización JSON.
- CPU: `VOYAGE_DEVICE=cpu`, puerto 9010, rate-limit desactivado (no afecta latencia).
- GPU: server ya cargado en 9009, sin reiniciar. Otras apps usan ~170 MiB de VRAM;
  la VRAM del server se mide por PID con `nvidia-smi`.
- Se hicieron 2 corridas por plataforma; la tabla usa la 2ª (variación entre corridas <5%,
  salvo `8 textos dim 2048` en CPU, ~13%).

## Resultados — GPU (RTX 3060, bf16)

| Caso | p50 cliente | p95 cliente | p50 server (`meta.ms`) | textos/s |
|---|---:|---:|---:|---:|
| 1 texto, dim 512 | 37.0 ms | 41.3 ms | 32 ms | 27.0 |
| 8 textos, dim 512 | 51.7 ms | 55.2 ms | 39 ms | 154.8 |
| 32 textos, dim 512 | 89.2 ms | 103.5 ms | 49 ms | 358.7 |
| 64 textos, dim 512 | 172.4 ms | 189.3 ms | 96 ms | 371.1 |
| 8 textos, dim 2048 | 78.2 ms | 91.2 ms | 41 ms | 102.3 |

## Resultados — CPU (Xeon E5-2680 v3, float32, 12 threads)

| Caso | p50 cliente | p95 cliente | p50 server (`meta.ms`) | textos/s |
|---|---:|---:|---:|---:|
| 1 texto, dim 512 | 69.7 ms | 81.2 ms | 64 ms | 14.3 |
| 8 textos, dim 512 | 233.8 ms | 276.9 ms | 224 ms | 34.2 |
| 32 textos, dim 512 | 1053.3 ms | 1184.5 ms | 1014 ms | 30.4 |
| 64 textos, dim 512 | 1766.8 ms | 1952.2 ms | 1696 ms | 36.2 |
| 8 textos, dim 2048 | 299.4 ms | 350.8 ms | 262 ms | 26.7 |

## Comparación directa

| Caso | GPU p50 | CPU p50 | Speedup end-to-end | GPU server | CPU server | Speedup inferencia |
|---|---:|---:|---:|---:|---:|---:|
| 1 texto, dim 512 | 37.0 ms | 69.7 ms | **1.9×** | 32 ms | 64 ms | 2.0× |
| 8 textos, dim 512 | 51.7 ms | 233.8 ms | **4.5×** | 39 ms | 224 ms | 5.7× |
| 32 textos, dim 512 | 89.2 ms | 1053.3 ms | **11.8×** | 49 ms | 1014 ms | **20.7×** |
| 64 textos, dim 512 | 172.4 ms | 1766.8 ms | **10.2×** | 96 ms | 1696 ms | **17.7×** |
| 8 textos, dim 2048 | 78.2 ms | 299.4 ms | **3.8×** | 41 ms | 262 ms | 6.4× |

Throughput (textos/s, p50 cliente):

| Caso | GPU | CPU | Speedup |
|---|---:|---:|---:|
| 1 texto, dim 512 | 27.0 | 14.3 | 1.9× |
| 8 textos, dim 512 | 154.8 | 34.2 | 4.5× |
| 32 textos, dim 512 | 358.7 | 30.4 | 11.8× |
| 64 textos, dim 512 | 371.1 | 36.2 | 10.2× |
| 8 textos, dim 2048 | 102.3 | 26.7 | 3.8× |

## Memoria

| | VRAM | RSS |
|---|---:|---:|
| GPU | **846 MiB** (784 MiB idle) | 2.05 GB |
| CPU | 0 MiB | 2.58 GB |

- La CPU usa `float32` (~1.3 GB de pesos), por eso su RSS es mayor.
- La GPU usa `bfloat16` (~660 MiB de pesos) + contexto CUDA y workspace de kernels.
- Tiempo de carga: CPU lista en **~5 s**; GPU no se midió en esta tanda (el server ya estaba arriba).
- Primera inferencia tras cargar el modelo en GPU: **~518 ms** (warmup de kernels + allocator);
  las siguientes bajan a **30–40 ms**. En CPU la primera request ronda **90–145 ms**.

## Lectura de resultados

1. **Lotes pequeños: gana GPU, pero poco (~2×).** El costo dominante es fijo
   (HTTP, tokenización, overhead del request), no el cómputo.
2. **Lotes grandes: la GPU se despega (~10–12× end-to-end, ~18–21× inferencia).**
   Ahí el paralelismo sí se aprovecha.
3. **La CPU no escala con el lote:** se queda en ~30–36 textos/s desde 8 textos.
   La GPU escala hasta ~371 textos/s con `internal_batch_size=32`.
4. **`truncate_dim` no reduce el cómputo:** el modelo siempre calcula 2048 dims y trunca
   al final. Por eso `8 textos dim 512` vs `dim 2048` tardan casi lo mismo en el server
   (39–41 ms GPU); la diferencia está en el payload de respuesta (8×2048 vs 8×512 floats).
5. **Serialización pesa:** con 64 textos dim 512, el cliente tarda ~76 ms más que el
   `meta.ms` del server en GPU (y ~71 ms en CPU). Ese delta es `.tolist()` + JSON +
   token stats, fuera del lock y fuera de `meta.ms`.
6. **Memoria:** la GPU consume ~2× menos RAM de proceso que la CPU y libera la CPU
   por completo; a cambio, ocupa ~846 MiB de VRAM.

## Reproducir

```bash
# GPU (server ya arriba)
BASE_URL=http://127.0.0.1:9009 env/bin/python scripts/verify/bench.py

# CPU (levantar un segundo server)
VOYAGE_DEVICE=cpu VOYAGE_PORT=9010 VOYAGE_RATE_LIMIT_PER_MINUTE=0 \
  VOYAGE_ATTN_IMPLEMENTATION=sdpa \
  env/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 9010

BASE_URL=http://127.0.0.1:9010 env/bin/python scripts/verify/bench.py
```

> `scripts/verify/bench.py` usa textos generados; los números de este documento salen de
> un harness equivalente con 8 textos fijos, así que los valores absolutos pueden variar
> un poco. Lo importante es la comparación relativa.

## Limitaciones

- No se midió concurrencia: el diseño es **1 request a la vez**; un segundo request recibe
  `503 gpu_busy` (sin cola).
- El benchmark es secuencial por cliente. No mide throughput agregado con varios clientes.
- La CPU compartía máquina con el server GPU y el escritorio; las cifras CPU pueden variar
  con la carga del sistema.
- `bfloat16` vs `float32` significa que la GPU además juega con ventaja de precisión/formato;
  es la configuración real, no un benchmark de silicio puro.
