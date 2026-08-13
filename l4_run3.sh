#!/usr/bin/env bash
# Solution 3 on the GPU box: physical-resolution normalisation, balanced
# leakage-free folds, fixed augmentation RNG, nested threshold selection.
#
#   gh auth login
#   curl -sL https://raw.githubusercontent.com/quanta-guy/uwe-ncc-void-detection/main/l4_run3.sh | bash
#
# The model and loss are solution 1's, unchanged - ten architectures were
# compared on this data and spanned 0.0245 Dice while folds spanned 0.15, so
# the model was never the binding constraint. Everything changed here is data
# handling and evaluation.
#
# Two stages. First five fold models, scored with nested tuning so the number
# involves no selection on the data it describes. Then three seeds trained on
# all 28 micrographs, which is the submission ensemble - those have no
# validation signal by construction, so they borrow the epoch count from the
# fold runs.
set -uo pipefail

ROOT="$HOME/uwe"
WORK="$ROOT/ai_hackathon_uwe_student"
LOG="$ROOT/solution3.log"
PY="$ROOT/venv/bin/python"
CODE_REPO="https://github.com/quanta-guy/uwe-ncc-void-detection"
DATA_REPO="https://github.com/KAngelov-NCC/ai_hackathon_uwe_student"
TAG="weights-s3"
EPOCHS="${EPOCHS:-30}"
STEPS="${STEPS:-200}"

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
# solution3 imports src/ for the U-Net and evaluation.py for scoring, so both
# trees have to land in the data clone.
cp -r "$ROOT/code/src" "$WORK/"
cp -r "$ROOT/code/solution3" "$WORK/"

say "ENV"
[ -d "$ROOT/venv" ] || python3 -m venv "$ROOT/venv" || {
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv "$ROOT/venv"; }
run "$PY" -m pip install -q --upgrade pip
run "$PY" -m pip install -q numpy scipy pandas pillow scikit-image opencv-python-headless
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "installing torch - ~2.5GB of CUDA wheels, several minutes, do not interrupt" | tee -a "$LOG"
  run "$PY" -m pip install torch
fi
"$PY" -c "import torch" 2>/dev/null || { echo "TORCH FAILED on $("$PY" -V)" | tee -a "$LOG"; exit 1; }
run "$PY" -c "import torch,cv2;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'cv2',cv2.__version__)"

cd "$WORK" || exit 1
W=$(( $(nproc) > 8 ? 8 : $(nproc) ))
mkdir -p solution3/runs

say "SMOKE  (fold balance, resampling, and that augmentation is no longer frozen)"
run "$PY" solution3/data3.py        || exit 1
run "$PY" solution3/train3.py --demo || exit 1

say "FOLDS 0-4  ($EPOCHS epochs x $STEPS steps, canonical 0.57 um/px)"
for f in 0 1 2 3 4; do
  [ -f "solution3/runs/s3_unet_f${f}_s0.pt" ] || run "$PY" solution3/train3.py \
      --epochs "$EPOCHS" --steps "$STEPS" --workers "$W" --fold "$f" --seed 0 --val-every 3 \
      --out "solution3/runs/s3_unet_f${f}_s0.pt"
done

say "NESTED SCORE  (knobs chosen on other folds, applied to the untouched one)"
run "$PY" solution3/evaluate3.py --oof --runs solution3/runs

say "SUBMISSION ENSEMBLE  3 seeds on all 28 micrographs"
for s in 0 1 2; do
  [ -f "solution3/runs/s3_unet_all_s${s}.pt" ] || run "$PY" solution3/train3.py \
      --epochs "$EPOCHS" --steps "$STEPS" --workers "$W" --fold -1 --seed "$s" \
      --out "solution3/runs/s3_unet_all_s${s}.pt"
done

say "PACKAGE"
ls -la solution3/runs/ | tee -a "$LOG"
tar czf "$ROOT/weights-s3.tgz" -C "$WORK" solution3/runs solution3/results
du -h "$ROOT/weights-s3.tgz" | tee -a "$LOG"

say "UPLOAD"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  cd "$ROOT/code" || exit 1
  gh release delete "$TAG" -y --cleanup-tag 2>/dev/null
  run gh release create "$TAG" "$ROOT/weights-s3.tgz" \
      -t "Solution 3 checkpoints" \
      -n "Physical-resolution normalisation, balanced leakage-free folds, fixed augmentation RNG, nested threshold selection. Trained on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1). Extract into the repo root: tar xzf weights-s3.tgz"
  echo "DOWNLOAD WITH:  gh release download $TAG -R quanta-guy/uwe-ncc-void-detection" | tee -a "$LOG"
else
  echo "gh not authenticated. Run 'gh auth login', then re-run - training is skipped, only upload repeats." | tee -a "$LOG"
  echo "Weights are at $ROOT/weights-s3.tgz" | tee -a "$LOG"
fi

say "DONE  $(date -u +%FT%TZ)"
echo "Full log: $LOG"
