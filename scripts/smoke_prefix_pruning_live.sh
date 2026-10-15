#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: BASE_URL=http://127.0.0.1:8000 [MODEL=model-id] ./scripts/smoke_prefix_pruning_live.sh

Optional environment:
  API_KEY               Bearer token for an authenticated server.
  SESSION_A / SESSION_B Session IDs used by the repeat requests.
  PROMPT_WORDS          Repeated prompt words (minimum 64, default 256).
  UNIQUE_REQUESTS       Distinct cache-fill requests (default 2).
  METRICS_WAIT_S        Delay before reading /metrics (default 1).
  CURL_MAX_TIME_S       Per-request HTTP timeout (default 30 seconds).

Run this against a server started separately with server.sh. It sends real
completion requests and checks prefix-cache metrics; it does not guarantee
cache eviction or CPU spill unless the server has sufficient cache pressure.
EOF
  exit 0
fi

cd "$ROOT"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-}"
API_KEY="${API_KEY:-}"
SESSION_A="${SESSION_A:-prefix-smoke-a}"
SESSION_B="${SESSION_B:-prefix-smoke-b}"
PROMPT_WORDS="${PROMPT_WORDS:-256}"
UNIQUE_REQUESTS="${UNIQUE_REQUESTS:-2}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi
if ! [[ "$PROMPT_WORDS" =~ ^[0-9]+$ ]] || ((PROMPT_WORDS < 64)); then
  echo "PROMPT_WORDS must be an integer >= 64 (got '$PROMPT_WORDS')." >&2
  exit 2
fi
if ! [[ "$UNIQUE_REQUESTS" =~ ^[0-9]+$ ]]; then
  echo "UNIQUE_REQUESTS must be a non-negative integer (got '$UNIQUE_REQUESTS')." >&2
  exit 2
fi

AUTH_ARGS=()
if [[ -n "$API_KEY" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer $API_KEY")
fi

get_json() {
  local url="$1"
  curl --silent --show-error --fail-with-body \
    --max-time "${CURL_MAX_TIME_S:-30}" \
    -H 'Accept: application/json' "${AUTH_ARGS[@]}" "$url"
}

post_completion() {
  local session_id="$1"
  local marker="$2"
  local payload response status body

  payload="$(python3 - "$MODEL" "$session_id" "$marker" "$PROMPT_WORDS" <<'PY'
import json
import sys

model, session_id, marker, words = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
prompt = f"Prefix pruning smoke test {marker}. " + "repeatable cache probe " * words
print(json.dumps({
    "model": model,
    "prompt": prompt,
    "session_id": session_id,
    "max_tokens": 1,
    "temperature": 0,
}))
PY
)"

  response="$(curl --silent --show-error \
    --max-time "${CURL_MAX_TIME_S:-30}" \
    -H 'Content-Type: application/json' \
    "${AUTH_ARGS[@]}" \
    --data-binary "$payload" \
    --write-out $'\n%{http_code}' \
    "$BASE_URL/v1/completions")"
  status="${response##*$'\n'}"
  body="${response%$'\n'*}"
  if [[ "$status" != 200 ]]; then
    printf 'Request failed (HTTP %s): %s\n' "$status" "$body" >&2
    return 1
  fi

  python3 -c '
import json, sys
data = json.loads(sys.stdin.read())
if not data.get("choices") or not isinstance(data["choices"][0].get("text"), str):
    raise SystemExit("Response did not contain a completion choice")
' <<<"$body"
}

metric_sum() {
  local metric="$1"
  python3 -c '
import re
import sys

metric = sys.argv[1]
total = 0.0
for line in sys.stdin:
    if not line or line.startswith("#"):
        continue
    name, _, value = line.partition(" ")
    if name == metric or name == metric + "_total" or name.startswith(metric + "{") or name.startswith(metric + "_total{"):
        try:
            total += float(value.strip())
        except ValueError:
            pass
print(total)
  ' "$metric"
}

echo "Checking vLLM at $BASE_URL"
models="$(get_json "$BASE_URL/v1/models")"
if [[ -z "$MODEL" ]]; then
  MODEL="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' <<<"$models")"
fi
if ! python3 -c '
import json
import sys
model = sys.argv[1]
if model not in {entry["id"] for entry in json.load(sys.stdin).get("data", [])}:
    raise SystemExit(f"Model '{model}' is not listed by this server")
  ' "$MODEL" <<<"$models"
then
  exit 1
fi
echo "Using model: $MODEL"

metrics_before="$(get_json "$BASE_URL/metrics" 2>/dev/null || true)"
gpu_hits_before="$(metric_sum vllm:prefix_cache_hits <<<"$metrics_before")"
cpu_hits_before="$(metric_sum vllm:external_prefix_cache_hits <<<"$metrics_before")"

echo "1/3 First request for session A"
post_completion "$SESSION_A" shared-prompt
echo "2/3 Repeat the same prompt/session (expect a prefix-cache hit)"
post_completion "$SESSION_A" shared-prompt

for ((i = 1; i <= UNIQUE_REQUESTS; i++)); do
  echo "Additional cache-fill request $i/$UNIQUE_REQUESTS"
  post_completion "prefix-smoke-fill-$i" "unique-fill-$i"
done

echo "3/3 Revisit the original prompt with session B"
post_completion "$SESSION_B" shared-prompt

sleep "${METRICS_WAIT_S:-1}"
metrics_after="$(get_json "$BASE_URL/metrics" 2>/dev/null || true)"
gpu_hits_after="$(metric_sum vllm:prefix_cache_hits <<<"$metrics_after")"
cpu_hits_after="$(metric_sum vllm:external_prefix_cache_hits <<<"$metrics_after")"

python3 - "$gpu_hits_before" "$gpu_hits_after" "$cpu_hits_before" "$cpu_hits_after" <<'PY'
import sys

gpu_before, gpu_after, cpu_before, cpu_after = map(float, sys.argv[1:])
gpu_delta = gpu_after - gpu_before
cpu_delta = cpu_after - cpu_before
print(f"GPU/local prefix-cache hit delta: {gpu_delta:g}")
print(f"CPU/offload prefix-cache hit delta: {cpu_delta:g}")
if gpu_delta < 1:
    raise SystemExit(
        "No local prefix-cache hit was reported. Check that prefix caching is "
        "enabled, the prompt exceeds one cache block, and /metrics is available."
    )
if cpu_delta < 1:
    print(
        "No CPU/offload hit was observed in this short run. This is normal "
        "unless GPU pressure caused blocks to spill to the CPU tier."
    )
PY

echo "Live API smoke test passed. This confirms request/session wiring and a cache hit;"
echo "it does not guarantee eviction or CPU spill without sufficient cache pressure."
