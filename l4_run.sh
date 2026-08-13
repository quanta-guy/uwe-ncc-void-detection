#!/usr/bin/env bash
# Solution 2 on the GPU box: albumentations pipeline, 3-class scratch U-Net,
# BCEWithLogits + Dice, then Hough-circle and KNN refinement at scoring time.
#
#   gh auth login
#   curl -sL https://raw.githubusercontent.com/quanta-guy/uwe-ncc-void-detection/main/l4_run.sh | bash
#
# Sits alongside l4_train.sh, which trains solution 1. This one trains six
# models: five folds on the full augmentation stack, plus one fold on the thin
# stack. That last run is the control - it is identical to the fold-0 model in
# every respect except the four added transforms, so the difference between
# them is what the augmentation fix is worth, measured rather than argued.
#
# Data comes from NCC's public repo, code from ours. The code repo carries no
# data and no evaluation.py on purpose: neither is ours to redistribute.
set -uo pipefail

ROOT="$HOME/uwe"
WORK="$ROOT/ai_hackathon_uwe_student"
LOG="$ROOT/solution2.log"
PY="$ROOT/venv/bin/python"
CODE_REPO="https://github.com/quanta-guy/uwe-ncc-void-detection"
DATA_REPO="https://github.com/KAngelov-NCC/ai_hackathon_uwe_student"
TAG="weights-s2"

mkdir -p "$ROOT"
say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "PROBE  $(date -u +%FT%TZ)"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 -VV; nproc; free -g | sed -n 2p; df -h "$HOME" | tail -1
} 2>&1 | tee -a "$LOG"

say "FETCH"
[ -d "$WORK/.git" ] || run git clone --depth 1 "$DATA_REPO" "$WORK"
if [ -d "$ROOT/code/.git" ]; then run git -C "$ROOT/code" fetch --all -q && run git -C "$ROOT/code" reset --hard origin/main
else run git clone "$CODE_REPO" "$ROOT/code"; fi
# solution2 imports src/ for the split, the U-Net and the scorer, so both trees
# have to land next to evaluation.py in the data clone.
cp -r "$ROOT/code/src" "$WORK/"
cp -r "$ROOT/code/solution2" "$WORK/"

say "ENV"
[ -d "$ROOT/venv" ] || python3 -m venv "$ROOT/venv" || {
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv "$ROOT/venv"; }
run "$PY" -m pip install -q --upgrade pip
run "$PY" -m pip install -q numpy scipy pandas pillow scikit-image
# NOT -q. torch drags in ~2.5GB of CUDA wheels and takes several minutes; with
# output suppressed it looks like a hang, which is an invitation to Ctrl-C it.
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "installing torch - ~2.5GB of CUDA wheels, several minutes, do not interrupt" | tee -a "$LOG"
  run "$PY" -m pip install torch
fi
"$PY" -c "import torch" 2>/dev/null || { echo "TORCH FAILED on $("$PY" -V)" | tee -a "$LOG"; exit 1; }
run "$PY" -m pip install segmentation_models_pytorch albumentations
# albumentations pulls opencv-python-headless; solution2 uses cv2 directly for
# Hough circles, so fail loudly here rather than an hour into the run.
run "$PY" -c "import cv2, albumentations as A; print('cv2', cv2.__version__, 'albumentations', A.__version__)" || exit 1
run "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

cd "$WORK" || exit 1
W=$(( $(nproc) > 8 ? 8 : $(nproc) ))
mkdir -p solution2/runs

say "PREFLIGHT"
run "$PY" src/preflight.py || { echo "environment incomplete - stopping before the long runs" | tee -a "$LOG"; exit 1; }

say "SMOKE"
run "$PY" solution2/pipeline.py     || exit 1
run "$PY" solution2/refine.py       || exit 1
run "$PY" solution2/train.py --demo || exit 1

say "FOLDS 0-4  full augmentation (scale jitter, colour gain, blur, noise, JPEG)"
for f in 0 1 2 3 4; do
  [ -f "solution2/runs/alb_unet_f$f.pt" ] || run "$PY" solution2/train.py \
      --epochs 20 --workers "$W" --fold "$f" --out "solution2/runs/alb_unet_f$f.pt"
done

say "CONTROL  fold 0 on the thin stack - isolates what the augmentation fix bought"
[ -f solution2/runs/alb_unet_thin_f0.pt ] || run "$PY" solution2/train.py \
    --epochs 20 --workers "$W" --fold 0 --aug thin --out solution2/runs/alb_unet_thin_f0.pt

# Scoring is CPU-bound severity geometry, so it normally runs on the laptop.
# Two folds are scored here anyway: enough to see whether the augmentation fix
# and the refinement worked before the weights are shipped home.
say "SCORE  fold 0, full vs thin, each with and without refinement"
run "$PY" solution2/evaluate.py --ckpt solution2/runs/alb_unet_f0.pt --refine
run "$PY" solution2/evaluate.py --ckpt solution2/runs/alb_unet_thin_f0.pt --refine

say "PACKAGE"
ls -la solution2/runs/ | tee -a "$LOG"
tar czf "$ROOT/weights-s2.tgz" -C "$WORK" solution2/runs solution2/results
du -h "$ROOT/weights-s2.tgz" | tee -a "$LOG"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import glob, torch
print(f"{'checkpoint':30} {'aug':6} {'norm':9} {'fold':>4} {'val_dice':>9}")
for p in sorted(glob.glob("solution2/runs/*.pt")):
    c = torch.load(p, map_location="cpu")
    print(f"{p.split('/')[-1]:30} {c.get('aug','full'):6} {c.get('norm','single'):9} "
          f"{c.get('fold',0):4d} {c.get('val_dice',float('nan')):9.4f}")
PY

say "UPLOAD"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  cd "$ROOT/code" || exit 1
  gh release delete "$TAG" -y --cleanup-tag 2>/dev/null
  run gh release create "$TAG" "$ROOT/weights-s2.tgz" \
      -t "Solution 2 checkpoints" \
      -n "Albumentations pipeline, 3-class scratch U-Net, BCE+Dice. Trained on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1). Extract into the repo root: tar xzf weights-s2.tgz"
  echo "DOWNLOAD WITH:  gh release download $TAG -R quanta-guy/uwe-ncc-void-detection" | tee -a "$LOG"
else
  echo "gh not authenticated. Run 'gh auth login', then re-run - training is skipped, only upload repeats." | tee -a "$LOG"
  echo "Weights are at $ROOT/weights-s2.tgz" | tee -a "$LOG"
fi

say "DONE  $(date -u +%FT%TZ)"
echo "Full log: $LOG"
