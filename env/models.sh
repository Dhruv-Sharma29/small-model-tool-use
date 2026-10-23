#!/usr/bin/env bash
# Models for M3: build the 14B GGUF and serve it, using project 01's pinned llama.cpp (tag b11439)
# and converter venv, so both models go through the identical pipeline and quantisation (Q4_K_M).
#
#   env/models.sh build-14b     # HF snapshot -> f16 GGUF -> Q4_K_M (f16 deleted afterwards)
#   env/models.sh serve-14b     # 127.0.0.1:21001, same flags as the 7B on :21000
#   env/models.sh stop-14b
#   env/models.sh serve GGUF PORT ALIAS   # e.g. a fine-tuned model; same flags
#   env/models.sh stop PORT
#
# The 7B (Qwen2.5-7B-Instruct a09a354, Q4_K_M) is served by 01 on 127.0.0.1:21000 with
# -c 32768 --parallel 4 (8192 tokens per slot); the 14B uses the same settings.
set -euo pipefail
cd "$(dirname "$0")/.."
L01=../01-llm-inference-lab
BIN=$L01/.tools/llama.cpp/build/bin
REV14=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8   # Qwen/Qwen2.5-14B-Instruct, Apache-2.0
SNAP="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/$REV14"
F16=models/qwen2.5-14b-instruct-cf98f3b-f16.gguf
Q4=models/qwen2.5-14b-instruct-cf98f3b-Q4_K_M.gguf
PORT=21001
PIDFILE=.run/llama-14b.pid
mkdir -p models .run/logs

running() { [ -f "$PIDFILE" ] && ps -o stat= -p "$(cat "$PIDFILE")" 2>/dev/null | grep -qv '^Z'; }

case "${1:-}" in
  build-14b)
    [ -d "$SNAP" ] || { echo "missing snapshot $SNAP (hf download Qwen/Qwen2.5-14B-Instruct --revision $REV14)" >&2; exit 1; }
    [ -f "$Q4" ] || {
      [ -f "$F16" ] || PYTHONPATH=$L01/.tools/llama.cpp/gguf-py $L01/.tools/convert-venv/bin/python \
        $L01/.tools/llama.cpp/convert_hf_to_gguf.py "$SNAP" --outtype f16 --outfile "$F16" > .run/logs/convert-14b.log 2>&1
      "$BIN/llama-quantize" "$F16" "$Q4" Q4_K_M > .run/logs/quantize-14b.log 2>&1
      rm -f "$F16"
    }
    sha256sum "$Q4" | tee -a models/SHA256SUMS ;;
  serve-14b)
    running && { echo "already running"; exit 0; }
    setsid "$BIN/llama-server" -m "$Q4" --alias qwen2.5-14b-instruct --host 127.0.0.1 --port $PORT \
      -ngl 99 -c 32768 --parallel 4 --jinja --metrics --slots > .run/logs/llama-14b.log 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 300); do
      curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q qwen2.5-14b && { echo "ready on :$PORT"; exit 0; }
      running || { echo "server exited; see .run/logs/llama-14b.log" >&2; exit 1; }
      sleep 1
    done
    echo "not ready after 300 s" >&2; exit 1 ;;
  serve)   # serve GGUF PORT ALIAS: any model, identical flags to the baselines
    gguf=$2; port=$3; alias=$4; pf=.run/llama-$port.pid
    [ -f "$pf" ] && ps -o stat= -p "$(cat "$pf")" 2>/dev/null | grep -qv '^Z' && { echo "port $port already served"; exit 0; }
    setsid "$BIN/llama-server" -m "$gguf" --alias "$alias" --host 127.0.0.1 --port "$port" \
      -ngl 99 -c 32768 --parallel 4 --jinja --metrics --slots > ".run/logs/llama-$port.log" 2>&1 &
    echo $! > "$pf"
    for _ in $(seq 1 300); do
      curl -fsS "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "$alias" && { echo "ready: $alias on :$port"; exit 0; }
      ps -o stat= -p "$(cat "$pf")" 2>/dev/null | grep -qv '^Z' || { echo "server exited; see .run/logs/llama-$port.log" >&2; exit 1; }
      sleep 1
    done
    echo "not ready after 300 s" >&2; exit 1 ;;
  stop)    # stop PORT
    pf=.run/llama-$2.pid; [ -f "$pf" ] && kill "$(cat "$pf")" 2>/dev/null || true; rm -f "$pf"; echo stopped ;;
  stop-14b)
    running && kill "$(cat "$PIDFILE")" || true; rm -f "$PIDFILE"; echo stopped ;;
  *) echo "usage: $0 build-14b | serve-14b | stop-14b | serve GGUF PORT ALIAS | stop PORT" >&2; exit 2 ;;
esac
