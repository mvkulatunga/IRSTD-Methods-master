#!/usr/bin/env bash
# Start a training run on the lab server from your laptop (see SERVER-GUIDE.md).
#
# Copies this working tree (uncommitted changes included) to the server with rsync, shows the
# GPU state, and starts main.py in a detached screen session named after the tag, logging to
# /srv/proj-mamba/runs/<your ID>/<tag>.log. The password is asked once: later ssh/rsync calls
# reuse the first connection for 10 minutes.
#
#   tools/remote_train.sh <tag> [main.py args]   push + start (default args: --profile r0-tree train)
#   tools/remote_train.sh log <tag>              follow a run's log (Ctrl-C stops following, not the run)
#   tools/remote_train.sh gpu                    GPU state
#   tools/remote_train.sh push                   copy the code only
#
# Examples:
#   tools/remote_train.sh tree-leah-1 --profile r0-tree train \
#       --stage1-from /srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth
#   tools/remote_train.sh base-leah-1 --profile r0 --model MOCIDBase train
#
# --tag is appended, so main.py args must include `train` (with --profile / --model before it).
# Override the defaults with MOCID_HOST, MOCID_REMOTE_DIR (repo on the server) or MOCID_PY.
set -euo pipefail

HOST="${MOCID_HOST:-a1851923@pl-lawr7615.services.adelaide.edu.au}"
REMOTE_DIR="${MOCID_REMOTE_DIR:-IRSTD-Methods-master}"   # relative to the server home
PY="${MOCID_PY:-/srv/proj-mamba/venv/bin/python}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# one shared connection, so the password is typed once
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=/tmp/mocid-ssh-%C" -o ControlPersist=10m
          -o ServerAliveInterval=60)
rssh() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }

push() {
    echo "[push] $REPO -> $HOST:~/$REMOTE_DIR"
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
        --exclude '__pycache__' --exclude 'MOCID/runs' --exclude '/venv' --exclude '/.venv' \
        "$REPO/" "$HOST:$REMOTE_DIR/"
}

gpu() {
    rssh 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
          echo "compute apps (pid, memory):"
          nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
}

case "${1:-}" in
    ""|-h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    gpu)  gpu; exit 0 ;;
    push) push; exit 0 ;;
    log)
        [ -n "${2:-}" ] || { echo "usage: $0 log <tag>"; exit 1; }
        rssh -t "tail -n 50 -F /srv/proj-mamba/runs/\$USER/$(printf %q "$2").log"
        exit 0 ;;
esac

TAG="$1"; shift
ARGS=("$@")
[ ${#ARGS[@]} -gt 0 ] || ARGS=(--profile r0-tree train)
[[ " ${ARGS[*]} " == *" train "* ]] || { echo "main.py args must include 'train'"; exit 1; }

push
echo "[gpu]"; gpu
util="$(rssh 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits' | head -1 | tr -d ' ')"
if [ "${util:-0}" -ge 50 ]; then
    read -r -p "GPU is ${util}% busy (someone else's job?). Start anyway? [y/N] " ok
    [[ "$ok" == [yY]* ]] || exit 1
fi

# the command screen runs, quoted for the remote shell
cmd="$(printf '%q ' "$PY" -u main.py "${ARGS[@]}" --tag "$TAG")"
log="/srv/proj-mamba/runs/\$USER/$TAG.log"
echo "[start] $cmd"
rssh "set -e
    export SCREENDIR=\$HOME/.screen; mkdir -p \$SCREENDIR; chmod 700 \$SCREENDIR
    if screen -ls | grep -q '[.]$TAG[[:space:]]'; then echo 'a screen session $TAG is already running'; exit 1; fi
    mkdir -p /srv/proj-mamba/runs/\$USER
    cd ~/$REMOTE_DIR/MOCID
    screen -dmS $(printf %q "$TAG") bash -c $(printf %q "$cmd 2>&1 | tee -a $log")
    sleep 2; screen -ls | grep '[.]$TAG[[:space:]]' || { echo 'screen exited at once; see the log'; tail -n 20 $log; exit 1; }"
echo "[ok] started '$TAG'. Follow with: $0 log $TAG   (attach on the server: screen -r $TAG)"
