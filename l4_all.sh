#!/usr/bin/env bash
# Everything, on one GPU box, then push the results back.
#
#   curl -sL https://raw.githubusercontent.com/quanta-guy/uwe-ncc-void-detection/main/l4_all.sh | bash
#
# Results, not weights, are what come home. The masks are ~50KB; the
# checkpoints are ~500MB and the only thing you do with a checkpoint is make
# masks, which this box can do itself. Set PUSH_WEIGHTS=1 to also upload the
# fold checkpoints as a release asset (outside git history).
#
# Needs `gh auth login` first if you want the push step to work. Without it
# everything still runs and lands in ~/uwe/out for manual retrieval.
set -uo pipefail

ROOT="$HOME/uwe"
WORK="$ROOT/ai_hackathon_uwe_student"
OUT="$ROOT/out"
LOG="$ROOT/all.log"
PY="$ROOT/venv/bin/python"
CODE_REPO="https://github.com/quanta-guy/uwe-ncc-void-detection"
DATA_REPO="https://github.com/KAngelov-NCC/ai_hackathon_uwe_student"
BRANCH="l4-results"

mkdir -p "$ROOT" "$OUT"
say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "PROBE  $(date -u +%FT%TZ)"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 -VV; nproc; free -g | sed -n 2p; df -h "$HOME" | tail -1
} 2>&1 | tee -a "$LOG"

say "FETCH"
[ -d "$WORK/.git" ] || run git clone --depth 1 "$DATA_REPO" "$WORK"
[ -d "$ROOT/code/.git" ] && run git -C "$ROOT/code" pull --ff-only || run git clone "$CODE_REPO" "$ROOT/code"
cp -r "$ROOT/code/src" "$WORK/"

say "ENV"
[ -d "$ROOT/venv" ] || python3 -m venv "$ROOT/venv" || {
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv "$ROOT/venv"; }
run "$PY" -m pip install -q --upgrade pip
run "$PY" -m pip install -q numpy scipy pandas pillow scikit-image
"$PY" -c "import torch" 2>/dev/null || run "$PY" -m pip install -q torch
"$PY" -c "import torch" 2>/dev/null || { echo "TORCH FAILED on $("$PY" -V)" | tee -a "$LOG"; exit 1; }
run "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

cd "$WORK" || exit 1
W=$(( $(nproc) > 8 ? 8 : $(nproc) ))

say "SMOKE"
for m in model.py train.py:--demo predict.py:--demo data.py; do
  f="${m%%:*}"; a="${m#*:}"; [ "$a" = "$f" ] && a=""
  run "$PY" "src/$f" $a || { echo "SMOKE FAILED at $f" | tee -a "$LOG"; exit 1; }
done

say "FOLDS 0-4"
for f in 0 1 2 3 4; do
  [ -f "runs/unet_f$f.pt" ] || run "$PY" src/train.py --epochs 20 --workers "$W" --fold "$f" --out "runs/unet_f$f.pt"
done

say "OUT-OF-FOLD SWEEP  (3100 images, each scored by the model that never saw it)"
run "$PY" src/predict.py --oof

say "DEPTH 3  vs  DEPTH 4"
[ -f runs/unet_d3.pt ] || run "$PY" src/train.py --epochs 20 --workers "$W" --depth 3 --fold 0 --out runs/unet_d3.pt
run "$PY" src/predict.py --tune --ckpt runs/unet_d3.pt
run "$PY" src/predict.py --tune --ckpt runs/unet_f0.pt

say "ARCHITECTURES  (falsification test: prediction is all land within ~0.02 Dice of 0.744)"
run "$PY" -m pip install -q segmentation_models_pytorch
for arch in unet_r34 unetpp_r34 unet_effb0 fpn_r34; do
  say "ARCH $arch"
  [ -f "runs/arch_$arch.pt" ] || run "$PY" src/train.py --arch "$arch" --epochs 20 --workers "$W" \
      --fold 0 --out "runs/arch_$arch.pt" || { echo "$arch FAILED - continuing" | tee -a "$LOG"; continue; }
  run "$PY" src/predict.py --tune --ckpt "runs/arch_$arch.pt"
done

say "SUBMISSION  (5-fold ensemble, best known knobs)"
ENS=(runs/unet_f0.pt runs/unet_f1.pt runs/unet_f2.pt runs/unet_f3.pt runs/unet_f4.pt)
run "$PY" src/predict.py --ckpt "${ENS[@]}" --threshold 0.3 --min-size 2
run "$PY" src/report.py --split test --ckpt "${ENS[@]}"

say "COLLECT"
rm -rf "$OUT"; mkdir -p "$OUT"
cp -r predicted_masks "$OUT/" 2>/dev/null
cp -r results "$OUT/" 2>/dev/null
cp "$LOG" "$OUT/run.log"
grep -E "^arch: |^best: |^best val Dice|^ground truth:|^  runs/" "$LOG" > "$OUT/metrics.txt" 2>/dev/null
ls -la runs/ > "$OUT/checkpoints.txt"
du -sh "$OUT" | tee -a "$LOG"

say "PUSH"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  cd "$ROOT/code" || exit 1
  git config user.email "hackathon@l4"; git config user.name "L4 runner"
  git checkout -B "$BRANCH" -q
  rm -rf predicted_masks results
  cp -r "$OUT"/* .
  git add -A && git commit -q -m "L4 results: 5-fold ensemble, OOF sweep, depth and architecture comparison

Generated on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1).
Masks and logs only - checkpoints stay on the box (see checkpoints.txt)."
  git push -q -u origin "$BRANCH" -f && echo "PUSHED to branch $BRANCH" | tee -a "$LOG"
  if [ "${PUSH_WEIGHTS:-0}" = "1" ]; then
    tar czf "$ROOT/folds.tgz" -C "$WORK" runs/unet_f0.pt runs/unet_f1.pt runs/unet_f2.pt runs/unet_f3.pt runs/unet_f4.pt
    gh release create l4-weights "$ROOT/folds.tgz" -t "Fold checkpoints" -n "5 fold models" 2>&1 | tail -2 | tee -a "$LOG"
  fi
else
  echo "gh not authenticated - results are in $OUT, run 'gh auth login' then re-run this script" | tee -a "$LOG"
fi

say "SUMMARY  $(date -u +%FT%TZ)"
cat "$OUT/metrics.txt" 2>/dev/null | tail -40
echo ""
echo "Full log: $LOG   Results: $OUT"
