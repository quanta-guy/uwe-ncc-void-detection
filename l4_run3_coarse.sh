#!/usr/bin/env bash
# Solution 3 at a COARSE canonical spacing, with the GPU verified in use.
#
#   TARGET_UM=1.33 bash l4_run3_coarse.sh
#   curl -sL https://raw.githubusercontent.com/quanta-guy/uwe-ncc-void-detection/main/l4_run3_coarse.sh | TARGET_UM=1.33 bash
#
# Why coarse. The 0.57um/px run was rejected: it misses visible voids on the
# Test set. The cause is measurable - only 28 of 3100 training images are
# natively at 0.57, so reaching that spacing UPSAMPLES 99.1% of training data
# (2.33x from the most common 1.33um/px, 3.5x from 2.00). Interpolation makes
# those textures soft, while Test images are natively 0.57 and sharp. The model
# learned voids in upsampled imagery and was asked about native imagery.
#
# Resampling DOWN to a coarse spacing inverts that: most training images are
# downsampled, which loses detail rather than inventing it, and the Test images
# get downsampled too - so both sides are treated the same way. That is the
# variable being tested. 1.33 is the modal training spacing, so it minimises
# how much resampling happens at all.
#
# On the GPU. Training is GPU-bound and takes ~7 minutes per fold. Scoring is
# NOT - evaluation.py's severity geometry is scipy convex hulls and KD-trees,
# so nvidia-smi correctly reads 0% during the NESTED SCORE stage. That is not a
# fault. This script asserts CUDA before training and samples utilisation
# throughout so the distinction is visible in the log.
set -uo pipefail

ROOT="$HOME/uwe"
WORK="$ROOT/ai_hackathon_uwe_student"
TARGET_UM="${TARGET_UM:-1.33}"
TAG_SUFFIX=$(echo "$TARGET_UM" | tr '.' 'p')
LOG="$ROOT/solution3_um${TAG_SUFFIX}.log"
GPULOG="$ROOT/gpu_um${TAG_SUFFIX}.log"
PY="$ROOT/venv/bin/python"
CODE_REPO="https://github.com/quanta-guy/uwe-ncc-void-detection"
DATA_REPO="https://github.com/KAngelov-NCC/ai_hackathon_uwe_student"
TAG="weights-s3-um${TAG_SUFFIX}"
EPOCHS="${EPOCHS:-30}"
STEPS="${STEPS:-200}"

mkdir -p "$ROOT"
say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "PROBE  $(date -u +%FT%TZ)   canonical spacing ${TARGET_UM} um/px"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 -VV; echo "nproc: $(nproc)"; free -g | sed -n 2p; df -h "$HOME" | tail -1
} 2>&1 | tee -a "$LOG"

say "FETCH"
[ -d "$WORK/.git" ] || run git clone --depth 1 "$DATA_REPO" "$WORK"
if [ -d "$ROOT/code/.git" ]; then run git -C "$ROOT/code" fetch --all -q && run git -C "$ROOT/code" reset --hard origin/main
else run git clone "$CODE_REPO" "$ROOT/code"; fi
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

# Hard gate. A CPU-only torch wheel trains without complaint and takes a day.
say "CUDA GATE"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import sys, torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda_available={ok}")
if ok:
    print(f"device: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"memory: {total/1024**3:.1f} GB total, {free/1024**3:.1f} GB free")
    x = torch.randn(4096, 4096, device="cuda")
    import time; torch.cuda.synchronize(); t = time.time()
    for _ in range(20):
        x = x @ x.T / 4096
    torch.cuda.synchronize()
    print(f"matmul check: 20x 4096^3 in {time.time()-t:.2f}s")
sys.exit(0 if ok else 1)
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "NO CUDA - refusing to train on CPU" | tee -a "$LOG"; exit 1; }

cd "$WORK" || exit 1
W=$(( $(nproc) > 8 ? 8 : $(nproc) ))
mkdir -p solution3/runs

# Sample GPU utilisation for the whole run so the training/scoring distinction
# is on the record rather than inferred from a single glance at nvidia-smi.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
           --format=csv,noheader -l 10 > "$GPULOG" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

say "SMOKE"
run "$PY" solution3/data3.py         || exit 1
run "$PY" solution3/train3.py --demo || exit 1

say "FOLDS 0-4  at ${TARGET_UM} um/px  ($EPOCHS epochs x $STEPS steps)"
for f in 0 1 2 3 4; do
  CK="solution3/runs/s3_um${TAG_SUFFIX}_f${f}_s0.pt"
  [ -f "$CK" ] || run "$PY" solution3/train3.py --require-cuda \
      --epochs "$EPOCHS" --steps "$STEPS" --workers "$W" --fold "$f" --seed 0 \
      --target-um "$TARGET_UM" --val-every 3 --out "$CK"
done

say "GPU DURING TRAINING  (should be well above 0%)"
tail -40 "$GPULOG" | tee -a "$LOG"

say "NESTED SCORE  - CPU-bound by design, nvidia-smi will read 0% here"
echo "evaluation.py's severity is scipy convex hulls and KD-trees. Not a fault." | tee -a "$LOG"
run "$PY" solution3/evaluate3.py --oof --runs solution3/runs \
    --pattern "s3_um${TAG_SUFFIX}_f{f}_s0.pt"

say "SUBMISSION ENSEMBLE  3 seeds on all 28 micrographs at ${TARGET_UM} um/px"
for s in 0 1 2; do
  CK="solution3/runs/s3_um${TAG_SUFFIX}_all_s${s}.pt"
  [ -f "$CK" ] || run "$PY" solution3/train3.py --require-cuda \
      --epochs "$EPOCHS" --steps "$STEPS" --workers "$W" --fold -1 --seed "$s" \
      --target-um "$TARGET_UM" --out "$CK"
done

say "PACKAGE"
ls -la solution3/runs/ | tee -a "$LOG"
cp "$GPULOG" solution3/results/ 2>/dev/null || true
tar czf "$ROOT/${TAG}.tgz" -C "$WORK" solution3/runs solution3/results
du -h "$ROOT/${TAG}.tgz" | tee -a "$LOG"

say "UPLOAD"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  cd "$ROOT/code" || exit 1
  gh release delete "$TAG" -y --cleanup-tag 2>/dev/null
  run gh release create "$TAG" "$ROOT/${TAG}.tgz" \
      -t "Solution 3 at ${TARGET_UM} um/px" \
      -n "Coarse canonical spacing, testing whether downsampling avoids the sharpness gap that sank the 0.57um/px run. Extract into the repo root."
  echo "DOWNLOAD WITH:  gh release download $TAG -R quanta-guy/uwe-ncc-void-detection" | tee -a "$LOG"
else
  echo "gh not authenticated. Weights are at $ROOT/${TAG}.tgz" | tee -a "$LOG"
fi

say "DONE  $(date -u +%FT%TZ)"
echo "log: $LOG"
echo "gpu utilisation trace: $GPULOG"
