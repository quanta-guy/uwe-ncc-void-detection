#!/usr/bin/env bash
# Solution 4: solution 1 with the augmentation RNG unfrozen. One variable.
#
#   curl -sL https://raw.githubusercontent.com/quanta-guy/uwe-ncc-void-detection/main/l4_run4.sh | bash
#
# src/data.py seeds its generator per item from torch.initial_seed(), which is
# a worker's INITIAL seed and never advances. Measured with num_workers=0, an
# image therefore receives exactly 1 distinct augmentation across every epoch -
# byte identical. The scale jitter, colour gain, blur, noise and JPEG re-encode
# that solution 1 was designed around have been drawn once and frozen.
#
# Everything else is held constant on purpose: same split, same _augment
# function imported from src/data.py, same model, same loss imported from
# src/train.py, same 20 epochs, same optimiser and schedule. The only
# difference is Micrographs4. So the difference in score is the RNG fix and
# nothing else.
#
# Both arms are also scored by the same non-nested OOF sweep that produced
# solution 1's 0.8869. That protocol is optimistic - solution 3 fixed it with
# nested selection - but both sides are equally optimistic, so the comparison
# stays fair. Mixing protocols would confound the variable under test.
#
# Three seeds per fold, because the measured run-to-run noise floor on this
# data is 0.0131 Dice and a single pair of runs cannot resolve a smaller
# difference than that.
set -uo pipefail

ROOT="$HOME/uwe"
WORK="$ROOT/ai_hackathon_uwe_student"
LOG="$ROOT/solution4.log"
GPULOG="$ROOT/gpu_s4.log"
PY="$ROOT/venv/bin/python"
CODE_REPO="https://github.com/quanta-guy/uwe-ncc-void-detection"
DATA_REPO="https://github.com/KAngelov-NCC/ai_hackathon_uwe_student"
TAG="weights-s4"
EPOCHS="${EPOCHS:-20}"
SEEDS="${SEEDS:-0 1 2}"

mkdir -p "$ROOT"
say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "PROBE  $(date -u +%FT%TZ)"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 -VV; echo "nproc: $(nproc)"; free -g | sed -n 2p; df -h "$HOME" | tail -1
} 2>&1 | tee -a "$LOG"

say "FETCH"
[ -d "$WORK/.git" ] || run git clone --depth 1 "$DATA_REPO" "$WORK"
if [ -d "$ROOT/code/.git" ]; then run git -C "$ROOT/code" fetch --all -q && run git -C "$ROOT/code" reset --hard origin/main
else run git clone "$CODE_REPO" "$ROOT/code"; fi
cp -r "$ROOT/code/src" "$WORK/"
cp -r "$ROOT/code/solution4" "$WORK/"

say "ENV"
[ -d "$ROOT/venv" ] || python3 -m venv "$ROOT/venv" || {
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv "$ROOT/venv"; }
run "$PY" -m pip install -q --upgrade pip
run "$PY" -m pip install -q numpy scipy pandas pillow scikit-image
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "installing torch - ~2.5GB of CUDA wheels, several minutes, do not interrupt" | tee -a "$LOG"
  run "$PY" -m pip install torch
fi

say "CUDA GATE"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import sys, time, torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda_available={ok}")
if ok:
    print(f"device: {torch.cuda.get_device_name(0)}")
    x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t = time.time()
    for _ in range(20): x = x @ x.T / 4096
    torch.cuda.synchronize(); print(f"matmul check: 20x 4096^3 in {time.time()-t:.2f}s")
sys.exit(0 if ok else 1)
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "NO CUDA - refusing to train on CPU" | tee -a "$LOG"; exit 1; }

cd "$WORK" || exit 1
W=$(( $(nproc) > 8 ? 8 : $(nproc) ))
mkdir -p solution4/runs

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
           --format=csv,noheader -l 10 > "$GPULOG" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

say "SMOKE  (must show solution 1 frozen at 1 augmentation, solution 4 fresh)"
run "$PY" solution4/data4.py         || exit 1
run "$PY" solution4/train4.py --demo || exit 1

for s in $SEEDS; do
  say "SEED $s  folds 0-4  ($EPOCHS epochs)"
  for f in 0 1 2 3 4; do
    CK="solution4/runs/s4_unet_f${f}_s${s}.pt"
    [ -f "$CK" ] || run "$PY" solution4/train4.py --require-cuda \
        --epochs "$EPOCHS" --workers "$W" --fold "$f" --seed "$s" --out "$CK"
  done
done

say "GPU DURING TRAINING  (should be well above 0%)"
tail -30 "$GPULOG" | tee -a "$LOG"

say "BASELINE  solution 1, same protocol - the number to beat"
run "$PY" solution4/evaluate4.py --runs runs --pattern "unet_f{f}.pt" --tag solution1_baseline

say "SOLUTION 4  one score per seed - scoring is CPU-bound, nvidia-smi reads 0% here"
for s in $SEEDS; do
  run "$PY" solution4/evaluate4.py --runs solution4/runs \
      --pattern "s4_unet_f{f}_s${s}.pt" --tag "solution4_seed${s}"
done

say "SUMMARY"
for d in solution4/results/*/best.txt; do
  echo "--- $d" | tee -a "$LOG"
  grep -E "final score|Dice_void|F2|TP/FP/FN" "$d" | tee -a "$LOG"
done
echo "" | tee -a "$LOG"
echo "Compare solution4_seed* against solution1_baseline. The measured noise" | tee -a "$LOG"
echo "floor is 0.0131 Dice - anything smaller than that is not a result." | tee -a "$LOG"

say "PACKAGE"
ls -la solution4/runs/ | tee -a "$LOG"
cp "$GPULOG" solution4/results/ 2>/dev/null || true
tar czf "$ROOT/weights-s4.tgz" -C "$WORK" solution4/runs solution4/results
du -h "$ROOT/weights-s4.tgz" | tee -a "$LOG"

say "UPLOAD"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  cd "$ROOT/code" || exit 1
  gh release delete "$TAG" -y --cleanup-tag 2>/dev/null
  run gh release create "$TAG" "$ROOT/weights-s4.tgz" \
      -t "Solution 4: solution 1 with the augmentation RNG fixed" \
      -n "One variable changed. src/data.py froze each image to a single augmentation because torch.initial_seed() never advances. Same split, augmentation, model, loss, schedule and scoring protocol as solution 1. Extract into the repo root."
  echo "DOWNLOAD WITH:  gh release download $TAG -R quanta-guy/uwe-ncc-void-detection" | tee -a "$LOG"
else
  echo "gh not authenticated. Weights are at $ROOT/weights-s4.tgz" | tee -a "$LOG"
fi

say "DONE  $(date -u +%FT%TZ)"
echo "log: $LOG"
echo "gpu trace: $GPULOG"
