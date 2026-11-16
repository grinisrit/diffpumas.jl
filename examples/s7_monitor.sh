#!/usr/bin/env bash
# Resource monitor and worker scaler of the S7 run (plain bash, no model). Usage:
#   examples/s7_monitor.sh OUT ORCHESTRATOR_PID [TARGET]
# Every 60 s appends one line to $OUT/resources.log:
#   utc_time  load1  host_cpu_pct  run_cpu_pct  run_rss_gb  avail_ram_gb  workers  wanted  paused
# and manages $OUT/WORKERS_WANTED and $OUT/PAUSE for examples/s7_protocol.sh:
#   load1 > 90 or available RAM < 48 GB  -> create PAUSE (no new units, no new workers)
#   load1 <= 85 and available RAM >= 52 GB -> remove PAUSE
#   load1 < 80 and available RAM > 64 GB and not paused -> WORKERS_WANTED += 3 up to TARGET (default 34)
# "run" = the process tree of the orchestrator. Exits when RUN_DONE or RUN_FAILED exists.
set -u
OUT="${1:?OUT}"; ORCH="${2:?orchestrator pid}"; TARGET="${3:-34}"
INTERVAL="${INTERVAL:-60}"
WANTED="$OUT/WORKERS_WANTED"; PAUSE="$OUT/PAUSE"; LOG="$OUT/resources.log"; PREV="$OUT/.monitor_prev"
: > "$PREV"
read_stat() { awk '/^cpu /{idle=$5+$6; tot=0; for(i=2;i<=NF;i++) tot+=$i; print tot, idle}' /proc/stat; }
read -r t0 i0 < <(read_stat); w0=$(date +%s.%N)
while :; do
  sleep "$INTERVAL"
  [ -e "$OUT/RUN_DONE" ] || [ -e "$OUT/RUN_FAILED" ] && { echo "$(date -u +%FT%TZ) monitor exit: run finished" >> "$LOG"; break; }
  read -r t1 i1 < <(read_stat); w1=$(date +%s.%N)
  host_cpu=$(awk -v t0="$t0" -v i0="$i0" -v t1="$t1" -v i1="$i1" 'BEGIN{d=t1-t0; printf "%.1f", (d>0 ? 100*(d-(i1-i0))/d : 0)}')
  dt=$(awk -v a="$w0" -v b="$w1" 'BEGIN{printf "%.3f", b-a}'); t0=$t1; i0=$i1; w0=$w1
  load1=$(cut -d' ' -f1 /proc/loadavg)
  avail=$(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo)
  # process tree of the orchestrator: pid ppid cumulative cpu seconds, rss kB, args
  read -r run_cpu run_rss workers < <(ps -eo pid=,ppid=,cputimes=,rss=,args= | awk -v root="$ORCH" -v dt="$dt" -v prevf="$PREV" '
    BEGIN { while ((getline l < prevf) > 0) { split(l, a, " "); prev[a[1]] = a[2] } close(prevf) }
    { pid[NR]=$1; par[$1]=$2; cpu[$1]=$3; rss[$1]=$4; line[$1]=$0 }
    END {
      inset[root]=1; changed=1
      while (changed) { changed=0; for (p in par) if (!(p in inset) && (par[p] in inset)) { inset[p]=1; changed=1 } }
      tot=0; mem=0; w=0; out=""
      for (p in inset) { if (!(p in cpu)) continue
        d = cpu[p] - ((p in prev) ? prev[p] : cpu[p]); if (d < 0) d = 0; tot += d
        mem += rss[p]; out = out p " " cpu[p] "\n"
        if (line[p] ~ /julia/ && line[p] ~ /lvd_tomography\.jl/ && line[p] ~ /--claim-dir/) w++ }
      printf "%s", out > prevf
      printf "%.1f %.2f %d\n", (dt > 0 ? 100*tot/dt : 0), mem/1048576, w }')
  wanted=$(cat "$WANTED" 2>/dev/null || echo 0); paused=0
  if awk -v l="$load1" -v a="$avail" 'BEGIN{exit !(l > 90 || a < 48)}'; then touch "$PAUSE"
  elif awk -v l="$load1" -v a="$avail" 'BEGIN{exit !(l <= 85 && a >= 52)}'; then rm -f "$PAUSE"; fi
  [ -e "$PAUSE" ] && paused=1
  if [ "$paused" = 0 ] && awk -v l="$load1" -v a="$avail" 'BEGIN{exit !(l < 80 && a > 64)}'; then
    wanted=$(( wanted + 3 > TARGET ? TARGET : wanted + 3 )); echo "$wanted" > "$WANTED"
  fi
  echo "$(date -u +%FT%TZ) $load1 $host_cpu $run_cpu $run_rss $avail $workers $wanted $paused" >> "$LOG"
done
