#!/usr/bin/env bash
# Smoke rápido: 3 curls para ver si la API respira. Uso: ./scripts/quick.sh [BASE_URL]
set -u
BASE="${1:-${BASE_URL:-http://127.0.0.1:9009}}"
echo "== BASE $BASE =="
curl -s "$BASE/health" | head -c 300; echo
curl -s "$BASE/ready" | head -c 300; echo
curl -s -X POST "$BASE/embed/query" -H 'Content-Type: application/json' \
  -d '{"texts":["Which planet is known as the Red Planet?"],"truncate_dim":512}' | head -c 500; echo
