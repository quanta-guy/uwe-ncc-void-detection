# Solution 3 — same model, rebuilt data and evaluation recipe

The model and loss are solution 1's, unchanged: scratch U-Net, weighted
cross-entropy plus soft Dice on the void channel. Ten architectures were
compared on this data and spanned **0.0245** Dice while folds spanned **0.15**.
The model was never the binding constraint, so nothing here touches it.

Everything changed is data handling and measurement. All four problems below
were verified against the data before any code was written.

## 1. Scale is magnification, not object size

```
fibre_radius_px x um_per_pixel  =  3.42 - 4.08 um   (median 3.99, CV 0.9%)
```

An 8 µm fibre diameter — standard carbon fibre. The 2–36 px spread of apparent
fibre radii is almost entirely microscope magnification, and the Test set sits
at 0.57 µm/px with radius 7 px, the same 4 µm product.

Solution 1 asked the network to learn invariance to this with 0.6–1.7× scale
jitter. Resampling every image to a canonical spacing removes the nuisance
instead — nnU-Net's target-spacing practice:

```
fibre radius after resampling to 0.57 um/px:  6.0 - 7.2 px   (was 2 - 36)
```

Inference resamples up, predicts, then resizes probabilities **back to native
resolution** before thresholding. Severity is measured in microns, so a mask on
the wrong grid reports the wrong severity for a void it located perfectly.

## 2. Validation counted near-duplicates

The index holds 1550 originals and 1550 stored `_aug_` copies. Every fold's
validation set was **exactly 50% augmented duplicates of originals already in
that fold** — 406 of 812 in fold 0. No train/val leakage, since micrograph
grouping is correct, but each case was counted twice and the hidden Test set
has no such structure. Validation here is originals only.

## 3. Folds were badly imbalanced

Striding sorted micrograph names gave:

```
images        812 / 676 / 638 / 428 / 546
void images   241 / 144 / 195 / 116 / 299     <- 21.3% vs 54.8% prevalence
```

Much of the reported 0.15 fold-to-fold spread is that imbalance rather than
genuine difficulty. A greedy allocator balances tiles, void and failing counts
proportionally while keeping every micrograph intact:

```
void images   105 / 109 /  92 / 106 / 110     spread 92% -> 17.2%
```

Note the first attempt used a lexicographic cost and put 20 of 28 micrographs
in one fold — a fold that falls behind on the first key stays the minimum
forever. Normalising each quantity by its dataset total fixes it.

## 4. Online augmentation was frozen

`data.py` seeded its generator from `(seed, index, torch.initial_seed())`.
`initial_seed()` is the worker's *initial* seed and never advances, so a given
image received the same transform every epoch. Measured with `num_workers=0`:

```
distinct augmentations per image over 6 epochs:   solution 1  1
                                                  solution 3  6
```

Much of solution 1's augmentation work was inert. The RNG here is created once
per worker and advances on every draw; `(seed, worker_id)` keeps runs
reproducible.

## Sampling

Micrographs contribute between a handful and several hundred tiles, so uniform
sampling trains mostly on the largest ones. Weights equalise micrographs
(`1/tiles_in_micrograph`) and boost void-containing tiles — nnU-Net's
foreground oversampling. Measured on a synthetic 90/10 split: void share 47%,
small-micrograph share 68%.

## Evaluation

**Nested threshold selection.** Solution 1's sweep chose threshold and
`min_size` on the same out-of-fold predictions it then reported as 0.8869.
Selecting on the data you report is optimistic by construction. Here the knobs
come from an inner split of the *other* folds and are applied unchanged to the
untouched outer fold.

**No ensemble scored on its own training data.** `bench.py` graded the 5-fold
ensemble on fold 0 where four of its five members had trained — that number was
invalid. Each outer fold is scored only by the model that never saw it; the
submission ensemble is separate seeds retrained on all 28 micrographs.

Because of this, **solution 3's number is not directly comparable to 0.8869**.
Compare it against a non-nested sweep of the same models.

## Run

```bash
python solution3/data3.py         # fold balance, resampling, RNG freshness
python solution3/train3.py --demo # loss falls, sampler actually rebalances

python solution3/train3.py --epochs 30 --fold 0
python solution3/evaluate3.py --oof --runs solution3/runs
```

`bash l4_run3.sh` does five folds, the nested score, and three all-data seeds
for the submission ensemble.

## Result: it does not ship, and the reason is worth more than the method

Nested out-of-fold across five folds:

| fold | threshold | min_size | Dice | F2 | final |
|---|---|---|---|---|---|
| 0 | 0.3 | 10 | 0.8051 | 0.9282 | 0.9282 |
| 1 | 0.3 | 10 | 0.7643 | 0.9810 | 0.9372 |
| 2 | 0.3 | 10 | 0.6045 | 0.9129 | 0.6899 |
| 3 | 0.3 | 10 | 0.7841 | 0.9611 | 0.9420 |
| 4 | 0.4 | 10 | 0.7456 | 0.9382 | 0.8744 |
| | | | | **mean** | **0.8743** (sd 0.096) |

Strong detection throughout - F2 0.91 to 0.98. Fold 2 collapses purely on the
Dice gate, which scales a healthy 0.9129 F2 down by 0.6045/0.8.

On the Test set it loses to solution 1: 4 FAIL against 6, missing two plainly
visible voids on `17-5-2_zoom_mid`. Not an operating-point problem - at
threshold 0.1 with min_size 2 it still detects nothing there.

**Why: the fix for scale introduced a sharpness nuisance.**

```
training images already at 0.57 um/px:   28 of 3100  (0.9%)
most common training spacing:            1.33 um/px  (1071 images)
test images:                             0.57 um/px, natively, all of them
```

Reaching canonical spacing means upsampling 99.1% of the training data - 2.33x
from 1.33um/px, 3.5x from 2.00um/px. Interpolation makes those textures soft.
The Test images are native and sharp. The model learned voids in upsampled
imagery and is asked about native imagery.

Its folds cannot see this, because every fold shares the same interpolation.
That is the same failure that made MicroNet look competitive on folds while it
collapsed on an unseen micrograph, arriving by a different route.

**The four defects found here are real and remain worth fixing** - the frozen
augmentation RNG especially, which made much of solution 1's augmentation
inert. What does not follow is that canonical resampling is the right response
to the scale spread. Downsampling to a coarse canonical spacing, or resampling
only the extremes, would test the idea without inverting the sharpness
distribution. That is the experiment to run next, not this one.
