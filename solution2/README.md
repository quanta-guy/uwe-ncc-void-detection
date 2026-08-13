# Solution 2 — albumentations + BCE/Dice + geometric refinement

Second attempt, kept separate from `src/` so the first one stays intact and the
two can be compared rather than merged.

| | solution 1 (`src/`) | solution 2 (here) |
|---|---|---|
| augmentation | hand-written | albumentations |
| model | scratch U-Net, 7.8M | scratch U-Net, 7.8M |
| head | 3-class softmax | 3-class multilabel |
| loss | weighted CE + soft Dice | `BCEWithLogitsLoss(pos_weight=[1,1,5])` + `smp.losses.DiceLoss` |
| normalisation | none (raw `[0,1]`) | in the Compose |
| post-process | threshold + despeckle | + Hough circles + KNN |

Split, optimiser, schedule, epochs and AMP are held constant on purpose.
Scoring goes through `predict._sweep`, which calls `evaluation.py` directly, so
the final number is comparable to `results/l4-2026-08-13/comparison.csv` line
for line.

## Run

```bash
python solution2/pipeline.py      # self-check: shapes, mask interpolation, determinism
python solution2/refine.py        # self-check: circles found, KNN vote, void-in-fibre suppressed
python solution2/train.py --demo  # self-check: loss falls, missing a void is expensive

python solution2/train.py --epochs 20 --fold 0
python solution2/evaluate.py --ckpt solution2/runs/alb_unet_f0.pt --refine
```

`bash l4_run.sh` does all five folds plus the control on a GPU box.

## What changed after the first attempt scored 0.8155

The first version of this pipeline used a thin augmentation stack, a binary
head and an ImageNet resnet34. It scored **0.8155** against **0.8446** for the
identical architecture in solution 1. Three fixes, in the order they were
ranked as suspects:

**1. Augmentation coverage.** Added scale jitter, per-channel colour gain,
blur, sensor noise and JPEG re-encode — each closing a gap that was *measured*,
not guessed. Fibre radius runs 2–22 px across micrographs and the Test set's
7 px is not among the nine radii in training; Test images are more saturated
(37 vs 25 mean channel spread); every Test image is JPEG while training is
nearly all TIFF. Blur, noise and compression are applied in that order because
that is the order a real capture applies them.

**2. 3-class target.** Binary discards the fibre/matrix distinction, and that
distinction is exactly what stops dark resin between fibres reading as a void.
The target is one-hot over three channels, so BCE still applies — multilabel
rather than binary.

**3. Scratch U-Net.** Removes the double-normalisation risk entirely: `build("unet")`
returns a bare model, so `A.Normalize` in the Compose is the only normalisation
in the system. It also drops a backbone that had already been measured to lose
here — every ImageNet encoder tried on this data underperformed the scratch net,
because natural-image priors do not transfer to micrographs.

With a scratch network the specified `A.Normalize(mean=(0.485,), std=(0.229,))`
is harmless — broadcasting one statistic across three channels is an affine
shift the first convolution undoes, and channel differences survive. It is the
default. `--norm imagenet` gives the proper 3-tuple.

## The control

`--aug thin` retrains with the original stack and nothing else changed, so the
augmentation fix is measured rather than assumed. `l4_run.sh` trains both.

## Refinement (`refine.py`)

Two classical steps, aimed at the one thing a per-pixel network cannot
represent: **a fibre is a circle of known size.**

**Hough circles.** `metadata.csv` gives `fibre_radius_px` per image, so the
transform gets a ±40% radius band instead of searching blind — a weak generic
detector becomes a strong one. The payoff is not a better fibre mask; the score
never reads the fibre class. It is that **voids do not occur inside fibres** —
porosity forms in the resin between them. A predicted void inside a confidently
detected fibre is a false positive, and deleting it raises precision without
touching a true void. The disc is shrunk to 75% of the detected radius, because
a void hugging the *outside* of a fibre is real.

**KNN.** The network's own confidence splits pixels into sure and unsure. The
sure ones are free labelled data — in this image, under this illumination. A
k-nearest-neighbour vote over colour, local mean, local texture and blue
chromaticity relabels the unsure ones from them. Per-image adaptation, no
training, no extra parameters. `scipy.spatial.cKDTree` does it; sklearn is not
a dependency.

Neither step is assumed to help. `evaluate.py --refine` scores with and without
and prints the delta — it ships only if it beats the plain sweep on held-out
data.

## Resolution

Everything stays at 256, the tiles' native size, so `A.Resize` is a no-op on
training data. This matters more than it looks: severity is measured in
**microns** via `um_per_pixel`, so a mask produced at 512 would report double
the true severity and fail every specimen. `evaluate.py` resizes probability
maps back to native resolution before thresholding, keeping that true for Test
images that are not 256 to begin with.
