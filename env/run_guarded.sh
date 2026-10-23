#!/usr/bin/env bash
# Run a long job in its own session, guarded by env/mem_guard.sh, and wait for it.
#   env/run_guarded.sh FLOOR_GB LOGFILE -- command args...
# The job's real PID is captured inside the new session (`echo $$; exec ...`): `setsid cmd &` can
# fork, which once left a guard watching a parent that had already exited (2026-10-09 pilot run).
set -euo pipefail
floor=$1; log=$2; shift 3
pidfile=$(mktemp)
setsid sh -c 'echo $$ > "$0"; exec "$@"' "$pidfile" "$@" > "$log" 2>&1 < /dev/null &
for _ in $(seq 50); do [ -s "$pidfile" ] && break; sleep 0.1; done
pid=$(cat "$pidfile"); rm -f "$pidfile"
"$(dirname "$0")/mem_guard.sh" "$floor" "$pid" > "$log.guard" 2>&1 &
echo "job pid $pid (guard log $log.guard)"
while ps -o stat= -p "$pid" 2>/dev/null | grep -qv '^Z'; do sleep 10; done
echo "job $pid ended"; cat "$log.guard"
