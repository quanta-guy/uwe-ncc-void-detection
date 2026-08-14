# Team Trojans — CFRP void detection

**Repository:** https://github.com/quanta-guy/uwe-ncc-void-detection
**Model:** 5-fold scratch U-Net ensemble · threshold 0.4 · min-size 4
**Out-of-fold final score:** 0.8869 (Dice_void 0.7562, F2 0.9383) — scored with
NCC's own `evaluation.py`, unmodified.

## Contents

- `predicted_masks/` — 32 PNGs for the Test set, class values 0 (matrix),
  1 (fibre), 2 (void)
- `SUBMISSION.md` — the model decision record: the four alternatives built and
  measured against this one, the measured noise floor, and known limits

## Reproduce

```bash
gh release download weights-submission --repo quanta-guy/uwe-ncc-void-detection
tar xzf unet_submission.tgz -C runs
python src/predict.py --ckpt runs/unet_f0.pt runs/unet_f1.pt runs/unet_f2.pt \
                             runs/unet_f3.pt runs/unet_f4.pt \
       --split test --out predicted_masks --threshold 0.4 --min-size 4
```

## Prototype UI

`frontend/` is a working inspection application over the same model: import
micrographs, run live inference, review fields by measured risk, correct masks,
export a cross-section report. `frontend/RUNNING.md` has the three-command start.
