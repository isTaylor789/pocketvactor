#!/usr/bin/env bash
# Suite de curls contra API viva (requiere modelo cargado).
# Uso: BASE_URL=http://127.0.0.1:9009 ./scripts/smoke.sh
# Cubre: health/ready/info, query+document happy path, TODAS las
# precisions (repro del bug bf16), TODOS los dims, validaciones 422,
# y regla SPEC-errors §1 (code == HTTP status).
set -u
BASE="${BASE_URL:-http://127.0.0.1:9009}"
PASS=0; FAIL=0
TMP=$(mktemp)

check() { # check <nombre> <codigo_esperado> <jq_condicion> <url> [curl_args...]
  local name="$1" want="$2" cond="$3" url="$4"; shift 4
  local code body ok="?"
  code=$(curl -s -o "$TMP" -w "%{http_code}" "$url" "$@")
  body=$(cat "$TMP")
  if [ "$code" != "$want" ]; then
    echo "FAIL $name — http=$code (esperaba $want) :: $(echo "$body" | head -c 220)"
    FAIL=$((FAIL+1)); return
  fi
  if command -v jq >/dev/null 2>&1; then
    if echo "$body" | jq -e "$cond" >/dev/null 2>&1; then ok="ok"; else ok="BAD-BODY"; fi
  else
    # sin jq: solo verifica code == body.code a ojo
    ok="no-jq"
  fi
  if [ "$ok" = "ok" ] || [ "$ok" = "no-jq" ]; then
    echo "PASS $name (http=$code)"
    PASS=$((PASS+1))
  else
    echo "FAIL $name — body no cumple [$cond] :: $(echo "$body" | head -c 300)"
    FAIL=$((FAIL+1))
  fi
}

J() { curl -s -X POST "$1" -H 'Content-Type: application/json' -d "$2"; }

echo "== $BASE =="

# --- sistema ---
check "health" 200 '.status=="success" and .code==200' "$BASE/health"
check "ready"  200 '.status=="success" and .code==200' "$BASE/ready"
check "info"   200 '.data.truncate_dims==[2048,1024,512,256]' "$BASE/info"

# --- happy path ---
check "query fp32/512" 200 '.status=="success" and .data.dim==512 and .data.count==1 and (.meta.truncated[0]==false)' \
  "$BASE/embed/query" -H 'Content-Type: application/json' \
  -d '{"texts":["Which planet is known as the Red Planet?"],"truncate_dim":512}'
check "document fp32/512" 200 '.status=="success" and .meta.endpoint=="document"' \
  "$BASE/embed/document" -H 'Content-Type: application/json' \
  -d '{"texts":["Mars, known for its reddish appearance, is often referred to as the Red Planet."],"truncate_dim":512}'

# --- precisions soportadas: TODAS deben dar 200, no 500 ---
for p in float32 binary ubinary; do
  check "query precision=$p (bug bf16)" 200 ".status==\"success\" and .data.precision==\"$p\"" \
    "$BASE/embed/query" -H 'Content-Type: application/json' \
    -d "{\"texts\":[\"hola mundo\"],\"truncate_dim\":512,\"precision\":\"$p\"}"
done
check "document precision=binary" 200 '.status=="success"' \
  "$BASE/embed/document" -H 'Content-Type: application/json' \
  -d '{"texts":["hola mundo"],"truncate_dim":512,"precision":"binary"}'

# --- dims MRL ---
for d in 2048 1024 512 256; do
  check "query dim=$d" 200 ".status==\"success\" and .data.dim==$d" \
    "$BASE/embed/query" -H 'Content-Type: application/json' \
    -d "{\"texts\":[\"test dim\"],\"truncate_dim\":$d}"
done

# --- orden + multi ---
check "orden 2 textos" 200 '.status=="success" and .data.count==2 and (.data.embeddings|length)==2' \
  "$BASE/embed/query" -H 'Content-Type: application/json' \
  -d '{"texts":["b","a"],"truncate_dim":256}'

# --- validaciones 422 (code == http) ---
check "lista vacia" 422 '.status=="error" and .code==422 and .error.details[0].reason=="empty_list"' \
  "$BASE/embed/query" -H 'Content-Type: application/json' -d '{"texts":[],"truncate_dim":512}'
check "whitespace+empty multi-indice" 422 '.error.details|length==2' \
  "$BASE/embed/query" -H 'Content-Type: application/json' -d '{"texts":["ok","   ",""],"truncate_dim":512}'
check "excede 32000" 422 '.error.details[0].reason=="exceeds_max_chars"' \
  "$BASE/embed/query" -H 'Content-Type: application/json' -d "{\"texts\":[\"$(python3 -c 'print("a"*32001)')\"],\"truncate_dim\":512}"
check "truncate_dim invalido + received" 422 '.error.type=="invalid_param" and .error.details[0].received==100' \
  "$BASE/embed/query" -H 'Content-Type: application/json' -d '{"texts":["ok"],"truncate_dim":100}'
check "precision invalida + received" 422 '.error.details[0].reason=="invalid_precision" and .error.details[0].received=="int4"' \
  "$BASE/embed/query" -H 'Content-Type: application/json' -d '{"texts":["ok"],"truncate_dim":512,"precision":"int4"}'
check "normalize string" 422 '.error.details[0].reason=="invalid_normalize"' \
  "$BASE/embed/query" -H 'Content-Type: application/json' -d '{"texts":["ok"],"truncate_dim":512,"normalize":"true"}'
check "falta truncate_dim" 422 '.error.details[0].field=="truncate_dim"' \
  "$BASE/embed/query" -H 'Content-Type: application/json' -d '{"texts":["hola"]}'
check "too_many received=65" 422 '.error.details[0].received==65' \
  "$BASE/embed/query" -H 'Content-Type: application/json' \
  -d "{\"texts\":[$(python3 -c 'print(",".join(["\"x\""]*65))')],\"truncate_dim\":512}"

# --- semaforo 1 sin cola: 2do concurrente -> 503 inmediato ---
python3 -c 'import json; t="Lorem ipsum dolor sit amet, consectetur adipiscing elit. "*40; print(json.dumps({"texts": [t]*64, "truncate_dim": 2048}))' > "$TMP.big"
curl -s -X POST "$BASE/embed/query" -H 'Content-Type: application/json' --data @"$TMP.big" > "$TMP.bg" &
BG=$!
sleep 0.5  # el 1ro (64 textos largos, dim 2048) debe seguir ocupando la GPU
check "concurrente rechazado 503 (sin cola)" 503 '.status=="error" and .code==503 and .error.type=="gpu_busy"' \
  "$BASE/embed/query" -H 'Content-Type: application/json' \
  -d '{"texts":["yo llego segundo"],"truncate_dim":512}'
wait $BG
if command -v jq >/dev/null 2>&1; then
  if jq -e '.status=="success"' "$TMP.bg" >/dev/null 2>&1; then echo "PASS primero completo (http=200)"; PASS=$((PASS+1));
  else echo "FAIL primero no completo :: $(head -c 200 "$TMP.bg")"; FAIL=$((FAIL+1)); fi
fi
rm -f "$TMP.bg" "$TMP.big"

# --- D6: mismo texto en query vs document -> vectores DISTINTOS (prompts) ---
cat > "$TMP.d6.py" << 'PYD6'
import json, sys, urllib.request
base = sys.argv[1]
txt = "Mars is known as the Red Planet."
def emb(path):
    req = urllib.request.Request(base + path,
        data=json.dumps({"texts": [txt], "truncate_dim": 512}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["data"]["embeddings"][0]
q, d = emb("/embed/query"), emb("/embed/document")
sys.exit(0 if q != d else 1)
PYD6
if python3 "$TMP.d6.py" "$BASE"; then
  echo "PASS D6 query!=document (prompts distintos)"; PASS=$((PASS+1))
else
  echo "FAIL D6 vectores iguales (prompts mezclados?)"; FAIL=$((FAIL+1))
fi
rm -f "$TMP.d6.py"

echo
echo "PASS=$PASS FAIL=$FAIL"
rm -f "$TMP"
[ "$FAIL" -eq 0 ]
