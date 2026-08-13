# Solution scaffold

U-Net → void probability → two tuned knobs (threshold, min blob size) → mask.
Four files in `src/`, nothing else.

## Run

```bash
conda activate uwe_hack
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series needs cu128+

python src/data.py            # self-checks: index, split, augmentation
python src/model.py
python src/train.py --demo
python src/predict.py --demo

python src/train.py --epochs 20 --out runs/unet_e20.pt
python src/predict.py --tune --ckpt runs/unet_e20.pt   # real score over a knob grid

# current best submission (3-model ensemble)
python src/predict.py --ckpt runs/unet_e20.pt runs/unet_e12.pt runs/unet.pt \
                      --threshold 0.3 --min-size 2
```

Smoke test the whole chain first: `python src/train.py --epochs 2 --limit 256`
then `python src/predict.py --tune --limit 60`. On an RTX 5060 a full epoch
over all 2288 training images is ~20 s, so 30 epochs is about 10 minutes —
budget for several full runs, not one.

## What the scaffold already decides

**Split by source micrograph, not by file.** The 4000 metadata rows are only
3100 distinct files (Data set II re-uses 250 images from Data set I), only
1550 originals (the rest are `_aug_N` transforms), and only **28 micrographs**
(the rest are tiles cut from them). Split any finer and every val image has a
near-copy — or a directly adjacent tile of the same slide — sitting in train.
The Test set is 2 entirely unseen micrographs, so `data.index_training()`
holds out every 5th micrograph and nothing below that level. That gives 2288
train / 812 val, of which 241 val images contain voids — enough to grade both
halves of the score.

**Images stay in colour.** Averaged over Data set I, void pixels read
(63, 59, 90) against matrix (177, 155, 161) — voids are not just darker, they
are *bluer*. Greyscaling throws that away for nothing.

**Four augmentations are added.** Flips, crops, brightness, contrast and CLAHE
are already baked into the "Augmented data set" folders. What is *not* covered
on disk:

- Every Test image is `.jpg`; nearly all training images are `.tif` → random JPEG re-encode.
- Test fibre radius is 7 px. The **original** data has 9 radii (2–6, 10, 11, 14, 22) and 7 is not one
  of them; the on-disk augmentation only reaches 7 as a side effect of a random resized crop that
  never zooms *out*, once per image → scale jitter over 0.6–1.7×, redrawn every epoch.
- Test images are more saturated than training ones (mean |R−G|+|G−B| of 37 vs 25) → per-channel colour gain.
- The on-disk Gaussian blur is applied **once and frozen**, and nothing on disk adds noise at all
  → defocus + sensor noise, redrawn each epoch, at p=0.3 each.

Applied in real capture order: geometry → colour → defocus → noise → JPEG.
Orientation is covered by the full dihedral group (`rot90` ×4 × hflip), exact
under `np.rot90` so labels never interpolate.

**Tuning happens against the real score, not a proxy.** `predict.py --tune`
imports `dice_void`, `compute_max_severity` and `compute_f2` from
`evaluation.py` and reports `F2 * min(1, Dice/0.8)` directly. Nothing is
re-derived, so a local win is a real win.

**The threshold is a knob because F2 is asymmetric.** A missed failure costs 4x
a false alarm, so the score-optimal void threshold is below the 0.5 an argmax
would use. Expect the sweep to land at 0.2–0.4.

## Score shape, and where the effort goes

```
final = F2 * min(1, mean_Dice_void / 0.8)
```

The Dice term is a **gate that saturates at 0.8** — past that, extra
segmentation quality buys literally nothing. F2 has no ceiling. So once the
val Dice clears ~0.8, stop improving the segmentation and spend the remaining
time on the pass/fail call.

Two facts worth exploiting there:

- Only **6 of 32** Test images contain voids (from `data/READ ME.txt`), and
  severity ≥ 25 is a *further* filter on those. The positive class is tiny, so
  a single flipped call moves F2 hard. Look at all 32 predictions by eye
  before submitting; that is 10 minutes and it is the highest-leverage review
  available.
- Fibre and matrix are **not scored at all**. They are predicted only because
  the submission format wants 0/1/2 and because the overlay makes a legible
  demo. Do not spend tuning time on them.

## Results so far

All on the 812-image / 6-micrograph val split.

| Setup | Dice | F2 | Final |
|---|---|---|---|
| single, 30-epoch schedule | 0.7287 | 0.9280 | 0.8453 |
| single, 20-epoch schedule | 0.7447 | 0.9286 | 0.8644 |
| **3-model ensemble** | **0.7442** | **0.9350** | **0.8698** |
| ensemble + TTA ×8 | 0.7435 | 0.9179 | 0.8556 |
| ensemble + dilate 1 px | 0.7028 | 0.9384 | 0.8244 |

Best: `--ckpt runs/unet_e20.pt runs/unet_e12.pt runs/unet.pt --threshold 0.3 --min-size 2`

### What was tried and rejected, with the reason

**Schedule length** — 30/12/20 epochs gave 0.7300/0.7428/0.7449 best Dice. The
spread (0.015) is smaller than the epoch-to-epoch noise *within* one run
(0.09). Not a real difference. Note the 30-epoch run peaked at epoch 7 while
still at ~94% of peak LR, then overfit for 23 epochs (train loss 1.10 → 0.41,
val flat).

**TTA over the 8 dihedral transforms** — *worse* (0.8556 vs 0.8644). Training
already augments over the full dihedral group, so the model is approximately
equivariant and averaging adds no information; it just smooths probability
maps, which hurts small voids. 45% of voids here are under 14 px across.

**Dilating predicted voids** — *worse*, and the reason is the important one.
The model recovers only 10–25% of void pixels on the specimens it misses, so
severity computed from our masks reads far below the same void's ground-truth
severity — and the fail line of 25 was calibrated on ground-truth masks.
Dilation corrects that and F2 does improve monotonically (FN 10 → 4), but Dice
collapses 0.744 → 0.590. **Because Dice is under 0.8 the gate is active, so
pixel accuracy cannot be traded for decision accuracy.** Above 0.8 this would
be free upside.

A scoring-side version of the same correction is not available: judging runs
`evaluation.py` against whatever mask we submit and applies its own line of
25. Any calibration has to live in the mask.

**Ensembling** — the only thing that helped (+0.005), and entirely through F2
(0.9286 → 0.9350) with Dice unchanged. Different minima fix *decisions*, not
*pixels*.

### The binding constraint

Dice is pinned at **0.744 ± 0.003** across every intervention tried. The gate
costs `1 − 0.744/0.8 = 7%` of the final score, and it also blocks the dilation
trade above. Everything left is downstream of moving Dice.

Two caveats on reading 0.8698:

- Val has 194 failures in 812 images (24%). The Test set has only 6
  void-containing images in 32. F2 on a failure-rich split does not transfer
  to a failure-sparse one.
- Ground truth is ImageJ circle-fitting plus threshold voids, which the README
  calls "not pixel-perfect". You cannot out-Dice your labels' self-consistency,
  and 0.8 may be roughly where that sits.

### Test-set sanity check

The submitted masks flag **6 of 32** images as void-containing — 2 in
`17-5-2_zoom_mid`, 4 in `2-6-1_mid`. `data/READ ME.txt` documents exactly 2
(C/LM-PEAK) and 4 (C/PEEK). Counts and split both match.

Most fragile call: `17-5-2_zoom_mid_512_512` at severity 25.7 against a line
of 25.

## Not built, and when to bother

- **ResNet34-UNet (`segmentation_models_pytorch`)** — the live option, and the
  only remaining lever on Dice. ~1 hour. Caveat: ImageNet stems are stride-4
  before any real processing, which is costly when the median void is 15 px.
- SegFormer-B0 — the only transformer whose tokenisation isn't disqualifying
  here (stride 4, not 14). DINOv2/SAM are not viable: 45% of voids are smaller
  than one 14 px patch.
- More val folds — 6 micrographs makes best-epoch selection partly luck.

## Submitting

```bash
git checkout -b submission/<team-name>
python src/predict.py --threshold <best> --min-size <best>
git add predicted_masks && git commit && git push -u origin submission/<team-name>
```

32 files, `<stem>.png`, values 0/1/2. Do not push to `main`.
