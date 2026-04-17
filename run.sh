#!/usr/bin/env bash
set -Eeuo pipefail

die() { echo "Error: $*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: ./run.sh <subcommand> [args...]

Subcommands:
  train_eval <cfg.yaml>   Run the train/eval pipeline.
                          <cfg.yaml> is a name in train_eval/yamls/ or an absolute path.
  gen_2d <cfg.cfg>        Run the 3D->2D synthetic image generator (Blender).
                          <cfg.cfg> is a name in image_gen_3d_to_2d/cfgs/ or an absolute path.

Behavior:
  - Runs under nohup; logs to logs/run_<TS>_<TAG>.log; logs/latest.log -> newest.
  - Streams logs to the terminal when interactive (Ctrl-C stops following only).
  - When the run finishes, waits SHUTDOWN_GRACE_SECS (default 120) and, if no other
    independent run.py is still alive, issues 'sudo -n shutdown now' (skipped if
    sudo would prompt for a password).

Environment:
  SHUTDOWN_GRACE_SECS   Seconds to wait before checking for siblings (default 120).
EOF
}

[[ $# -ge 1 ]] || { usage; exit 1; }
SUBCMD="$1"; shift

case "$SUBCMD" in
  train_eval)
    [[ $# -ge 1 ]] || die "train_eval requires a YAML config (e.g., ./run.sh train_eval main.yaml)"
    CFG_ARG="$1"
    case "$CFG_ARG" in
      *.yaml|*.yml) : ;;
      *) die "Config must be a .yaml/.yml file: $CFG_ARG" ;;
    esac
    TAG="$(basename "${CFG_ARG%.*}")"
    PY_ARGS=(run.py train_eval --cfg_file "$CFG_ARG")
    ;;
  gen_2d)
    [[ $# -ge 1 ]] || die "gen_2d requires a CFG config (e.g., ./run.sh gen_2d dr.cfg)"
    CFG_ARG="$1"
    case "$CFG_ARG" in
      *.cfg) : ;;
      *) die "Config must be a .cfg file: $CFG_ARG" ;;
    esac
    TAG="gen_2d_$(basename "${CFG_ARG%.*}")"
    PY_ARGS=(run.py gen_2d --cfg "$CFG_ARG")
    ;;
  -h|--help)
    usage; exit 0
    ;;
  *)
    die "Unknown subcommand: $SUBCMD (try ./run.sh --help)"
    ;;
esac

SHUTDOWN_GRACE_SECS="${SHUTDOWN_GRACE_SECS:-120}"

# --- enter repo root (directory containing this script) ---
cd "$(dirname "$0")"

[[ -x ".venv/bin/python" ]] || die ".venv not found. Run ./scripts/setup.sh first."

export PYTHONPATH="."

# --- update repo (best-effort) ---
git pull --rebase --autostash || echo "git pull failed (continuing anyway)"

# --- logging setup ---
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/run_${TS}_${TAG}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"

# --- start under nohup ---
echo "Starting: ./.venv/bin/python ${PY_ARGS[*]}"
nohup ./.venv/bin/python "${PY_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
PY_PID=$!
echo "$PY_PID" > "$LOG_DIR/run.pid"

echo "run.py PID : $PY_PID"
echo "Log file   : $LOG_FILE"
echo "Latest log : $LOG_DIR/latest.log"
echo

# --- detached watcher: after run.py exits, wait, then shutdown if no other run.py is alive ---
nohup bash -c '
  set -Eeuo pipefail
  py_pid="$1"
  log_file="$2"
  grace_secs="$3"

  descendants() {
    local pid="$1"
    local kids
    kids=$(pgrep -P "$pid" || true)
    for k in $kids; do
      echo "$k"
      descendants "$k"
    done
  }

  while kill -0 "$py_pid" 2>/dev/null; do sleep 5; done
  sleep "$grace_secs"

  mapfile -t all_runs < <(pgrep -u "$USER" -f "[r]un\.py" || true)
  mapfile -t child_pids < <(descendants "$py_pid" | sort -u)

  in_children() {
    local needle="$1"
    for c in "${child_pids[@]}"; do
      [[ "$needle" -eq "$c" ]] && return 0
    done
    return 1
  }

  others_exist=0
  for p in "${all_runs[@]}"; do
    [[ "$p" -eq "$py_pid" ]] && continue
    if ! in_children "$p"; then
      others_exist=1
      break
    fi
  done

  if [[ "$others_exist" -eq 0 ]]; then
    printf "%s  run.py finished; no other independent run.py instances; issuing shutdown.\n" "$(date)" >> "$log_file"
    sudo -n shutdown now || {
      printf "%s  sudo shutdown failed or needed a password; skipping shutdown.\n" "$(date)" >> "$log_file"
    }
  else
    keepers=$(printf "%s\n" "${all_runs[@]}" | tr "\n" " ")
    printf "%s  Another independent run.py is still running; not shutting down. PIDs: %s\n" \
      "$(date)" "$keepers" >> "$log_file"
  fi
' _ "$PY_PID" "$LOG_FILE" "$SHUTDOWN_GRACE_SECS" >/dev/null 2>&1 &

# --- live log streaming if interactive ---
if [ -t 1 ]; then
  echo "Streaming logs. Press Ctrl-C to stop following (run continues in background)."
  echo "Tip: run 'tail -f $LOG_FILE' anytime."
  tail -n +1 -f "$LOG_FILE" &
  TAIL_PID=$!
  wait "$PY_PID" || true
  kill "$TAIL_PID" >/dev/null 2>&1 || true
  echo
  echo "run.py exited. See full logs in: $LOG_FILE"
else
  echo "Non-interactive session detected. Run continues under nohup."
  echo "Check progress later with: tail -f $LOG_FILE"
fi
