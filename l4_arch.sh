#!/usr/bin/env bash
# Architecture comparison. Run AFTER l4_run.sh - it reuses that venv and data.
#
#   curl -sL https://raw.githubusercontent.com/quanta-guy/uwe-ncc-void-detection/main/l4_arch.sh | bash
#
# This is a falsification test, not a leaderboard. The prediction, stated
# before running: every architecture lands within ~0.02 Dice of the scratch
# U-Net's 0.744, because the ceiling is label noise rather than model capacity.
# A null result here is the useful result - it turns "we think the labels cap
# us" into a measured claim. If something clears 0.80, that prediction was
# wrong and we swap it in.
#
# Everything except the architecture is held fixed: same fold, same epochs,
# same loss, same schedule, same augmentation.
set -uo pipefail

ROOT="$HOME/uwe"
WORK="$ROOT/ai_hackathon_uwe_student"
LOG="$ROOT/arch.log"
PY="$ROOT/venv/bin/python"

[ -x "$PY" ] || { echo "run l4_run.sh first (no venv at $PY)"; exit 1; }
cd "$WORK" || exit 1

say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "REFRESH CODE"
run git -C "$ROOT/code" pull --ff-only
cp -r "$ROOT/code/src" "$WORK/"

say "INSTALL segmentation_models_pytorch"
run "$PY" -m pip install -q segmentation_models_pytorch
run "$PY" -c "import segmentation_models_pytorch as smp; print('smp', smp.__version__)"

W=$(( $(nproc) > 8 ? 8 : $(nproc) ))

for arch in unet_r34 unetpp_r34 unet_effb0 fpn_r34; do
  say "ARCH $arch"
  if [ ! -f "runs/arch_$arch.pt" ]; then
    run "$PY" src/train.py --arch "$arch" --epochs 20 --workers "$W" \
        --fold 0 --out "runs/arch_$arch.pt" || { echo "$arch FAILED" | tee -a "$LOG"; continue; }
  fi
  run "$PY" src/predict.py --tune --ckpt "runs/arch_$arch.pt"
done

say "SCRATCH U-NET BASELINE (same fold, for comparison)"
run "$PY" src/predict.py --tune --ckpt runs/unet_f0.pt

say "COMPARISON"
echo "" | tee -a "$LOG"
printf '%-14s %-9s %s\n' "arch" "params" "best" | tee -a "$LOG"
awk '
  /^arch: /            { a=$2; p=$4 }
  /^  runs\/arch_/     { split($1,q,"/"); sub(/\.pt$/,"",q[2]); a=q[2]; sub(/^arch_/,"",a) }
  /^  runs\/unet_f0/   { a="unet (scratch)" }
  /^best: /            { print a "|" p "|" $0 }
' "$LOG" | awk -F'|' '{ printf "%-14s %-9s %s\n", $1, $2, $3 }' | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Full log: $LOG"
