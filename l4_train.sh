#!/usr/bin/env bash
# Train every model on the GPU box, then ship the weights home.
#
#   gh auth login
#   curl -sL https://raw.githubusercontent.com/quanta-guy/uwe-ncc-void-detection/main/l4_train.sh | bash
#
# Training only. Scoring is CPU-bound severity geometry rather than matrix
# multiplication, so it runs on the laptop via src/bench.py, which writes one
# results directory per model.
#
# Weights go up as a release asset, not a git commit: several are near or over
# GitHub's 100MB per-file push limit, and binaries in git history are forever.
set -uo pipefail

ROOT="$HOME/uwe"
WORK="$ROOT/ai_hackathon_uwe_student"
LOG="$ROOT/train.log"
PY="$ROOT/venv/bin/python"
CODE_REPO="https://github.com/quanta-guy/uwe-ncc-void-detection"
DATA_REPO="https://github.com/KAngelov-NCC/ai_hackathon_uwe_student"
TAG="weights"

mkdir -p "$ROOT"
say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "PROBE  $(date -u +%FT%TZ)"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 -VV; nproc; free -g | sed -n 2p; df -h "$HOME" | tail -1
} 2>&1 | tee -a "$LOG"

say "FETCH"
[ -d "$WORK/.git" ] || run git clone --depth 1 "$DATA_REPO" "$WORK"
if [ -d "$ROOT/code/.git" ]; then run git -C "$ROOT/code" pull --ff-only
else run git clone "$CODE_REPO" "$ROOT/code"; fi
cp -r "$ROOT/code/src" "$WORK/"

say "ENV"
[ -d "$ROOT/venv" ] || python3 -m venv "$ROOT/venv" || {
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv "$ROOT/venv"; }
run "$PY" -m pip install -q --upgrade pip
run "$PY" -m pip install -q numpy scipy pandas pillow scikit-image
"$PY" -c "import torch" 2>/dev/null || run "$PY" -m pip install -q torch
"$PY" -c "import torch" 2>/dev/null || { echo "TORCH FAILED on $("$PY" -V)" | tee -a "$LOG"; exit 1; }
run "$PY" -m pip install -q segmentation_models_pytorch
run "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

cd "$WORK" || exit 1
W=$(( $(nproc) > 8 ? 8 : $(nproc) ))
mkdir -p runs

say "SMOKE"
run "$PY" src/model.py          || exit 1
run "$PY" src/train.py --demo   || exit 1
run "$PY" src/predict.py --demo || exit 1
run "$PY" src/data.py           || exit 1

say "FOLDS 0-4  (micrograph-level holdout, 20 epochs each)"
for f in 0 1 2 3 4; do
  [ -f "runs/unet_f$f.pt" ] || run "$PY" src/train.py --epochs 20 --workers "$W" \
      --fold "$f" --out "runs/unet_f$f.pt"
done

say "DEPTH 3  (median void is 15px; at depth 4 that is sub-pixel at the bottleneck)"
[ -f runs/unet_d3.pt ] || run "$PY" src/train.py --epochs 20 --workers "$W" \
    --depth 3 --fold 0 --out runs/unet_d3.pt

say "ARCHITECTURES  (prediction on record: all land within ~0.02 Dice of 0.744)"
for arch in unet_r34 unetpp_r34 unet_effb0 fpn_r34; do
  [ -f "runs/arch_$arch.pt" ] && continue
  run "$PY" src/train.py --arch "$arch" --epochs 20 --workers "$W" \
      --fold 0 --out "runs/arch_$arch.pt" \
    || echo "$arch FAILED - continuing" | tee -a "$LOG"
done

say "PACKAGE"
ls -la runs/ | tee -a "$LOG"
tar czf "$ROOT/weights.tgz" -C "$WORK" runs
du -h "$ROOT/weights.tgz" | tee -a "$LOG"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import glob, torch
print(f"{'checkpoint':28} {'arch':14} {'depth':>5} {'fold':>4} {'val_dice':>9}")
for p in sorted(glob.glob("runs/*.pt")):
    c = torch.load(p, map_location="cpu")
    print(f"{p.split('/')[-1]:28} {c.get('arch','unet'):14} {c.get('depth',4):5d} "
          f"{c.get('fold',0):4d} {c.get('val_dice',float('nan')):9.4f}")
PY

say "UPLOAD"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  cd "$ROOT/code" || exit 1
  gh release delete "$TAG" -y --cleanup-tag 2>/dev/null
  run gh release create "$TAG" "$ROOT/weights.tgz" \
      -t "Trained checkpoints" \
      -n "All models trained on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1). Extract into the repo root: tar xzf weights.tgz"
  echo "DOWNLOAD WITH:  gh release download $TAG -R quanta-guy/uwe-ncc-void-detection" | tee -a "$LOG"
else
  echo "gh not authenticated. Run 'gh auth login', then re-run - training is skipped, only upload repeats." | tee -a "$LOG"
  echo "Weights are at $ROOT/weights.tgz" | tee -a "$LOG"
fi

say "DONE  $(date -u +%FT%TZ)"
echo "Full log: $LOG"
