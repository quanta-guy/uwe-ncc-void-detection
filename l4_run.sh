#!/usr/bin/env bash
# Full experiment queue for a fresh GPU box. Idempotent: safe to re-run.
#
#   curl -sL https://raw.githubusercontent.com/quanta-guy/uwe-ncc-void-detection/main/l4_run.sh | bash
#
# Data comes from the public challenge repo, code from ours, so nothing needs
# uploading. Everything lands in ~/uwe and every step appends to ~/uwe/run.log.
set -uo pipefail

ROOT="$HOME/uwe"
WORK="$ROOT/ai_hackathon_uwe_student"
LOG="$ROOT/run.log"
CODE_REPO="https://github.com/quanta-guy/uwe-ncc-void-detection"
DATA_REPO="https://github.com/KAngelov-NCC/ai_hackathon_uwe_student"

mkdir -p "$ROOT"
say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "PROBE  $(date -u +%FT%TZ)"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 --version; nproc; free -g | head -2; df -h "$HOME" | tail -1
} 2>&1 | tee -a "$LOG"

say "FETCH"
[ -d "$WORK/.git" ] || run git clone --depth 1 "$DATA_REPO" "$WORK"
run git -C "$WORK" pull --ff-only
rm -rf "$ROOT/code"
run git clone --depth 1 "$CODE_REPO" "$ROOT/code"
cp -r "$ROOT/code/src" "$WORK/"
echo "data files: $(find "$WORK/data" -type f | wc -l)" | tee -a "$LOG"

say "ENV"
VENV="$ROOT/venv"
[ -d "$VENV" ] || python3 -m venv "$VENV" || { sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv "$VENV"; }
PY="$VENV/bin/python"
run "$PY" -m pip install -q --upgrade pip
run "$PY" -m pip install -q numpy scipy pandas pillow scikit-image
# Plain PyPI wheels already bundle CUDA 12.x and cover a far wider Python
# range than the pinned cu124 index, which only carries torch 2.4-2.6 and
# fails with ResolutionImpossible on newer interpreters.
run "$PY" -m pip install -q torch || run "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121
"$PY" -c "import torch" 2>/dev/null || { echo "TORCH INSTALL FAILED - python is $("$PY" -V)" | tee -a "$LOG"; exit 1; }
run "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

cd "$WORK" || exit 1

say "SMOKE"
run "$PY" src/model.py    || exit 1
run "$PY" src/train.py --demo   || exit 1
run "$PY" src/predict.py --demo || exit 1
run "$PY" src/data.py     || exit 1

# Workers: dataloader is PIL-bound, so scale with cores but do not oversubscribe.
W=$(( $(nproc) > 8 ? 8 : $(nproc) ))

say "FOLDS 0-4  (20 epochs each, micrograph-level holdout)"
for f in 0 1 2 3 4; do
  [ -f "runs/unet_f$f.pt" ] && { echo "fold $f already done" | tee -a "$LOG"; continue; }
  run "$PY" src/train.py --epochs 20 --workers "$W" --fold "$f" --out "runs/unet_f$f.pt"
done

say "OUT-OF-FOLD SWEEP  (all 3100 images, each scored by the model that never saw it)"
run "$PY" src/predict.py --oof

say "DEPTH 3  (median void is 15px; at depth 4 that is sub-pixel at the bottleneck)"
[ -f runs/unet_d3.pt ] || run "$PY" src/train.py --epochs 20 --workers "$W" --depth 3 --out runs/unet_d3.pt
run "$PY" src/predict.py --tune --ckpt runs/unet_d3.pt

say "BASELINE FOR COMPARISON  (depth 4, same fold)"
run "$PY" src/predict.py --tune --ckpt runs/unet_f0.pt

say "DONE  $(date -u +%FT%TZ)"
echo "" | tee -a "$LOG"
echo "SUMMARY -----------------------------------------------------" | tee -a "$LOG"
grep -E "^best:|^best val|torch |NVIDIA|final score" "$LOG" | tail -30
echo "-------------------------------------------------------------"
echo "Full log: $LOG"
