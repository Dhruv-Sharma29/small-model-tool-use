#!/usr/bin/env bash
# Stop OUR processes (given PIDs) if the shared machine's available memory drops below a floor, so a
# long evaluation never pushes other users' jobs into the OOM killer. Logs and exits when it fires
# or when all watched PIDs are gone.   usage: env/mem_guard.sh FLOOR_GB PID [PID ...]
# Start each watched job with `setsid` so its PID is also its process-group id.
floor_kb=$(( $1 * 1048576 )); shift
while :; do
  alive=0; for p in "$@"; do ps -o stat= -p "$p" 2>/dev/null | grep -qv '^Z' && alive=1; done
  [ $alive = 0 ] && { echo "$(date -u +%FT%TZ) all watched processes gone"; exit 0; }
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$avail" -lt "$floor_kb" ]; then
    echo "$(date -u +%FT%TZ) MemAvailable $((avail/1048576)) GB < $(( floor_kb/1048576 )) GB: stopping $*"
    # Each watched PID should lead its own process group (start runners with setsid), so the whole
    # job goes, not just a wrapper shell. Falls back to the single PID.
    for p in "$@"; do kill -- "-$p" 2>/dev/null || kill "$p" 2>/dev/null; done; exit 1
  fi
  sleep 15
done
