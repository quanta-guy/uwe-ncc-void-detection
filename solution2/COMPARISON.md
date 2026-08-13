# Solution 2 vs solution 1 — fold 0, 812 held-out images, 6 micrographs

All scored by `predict._sweep` calling `evaluation.py`, so these are the
challenge's own numbers computed identically. Ground truth on this fold: 241
void-containing images, 194 failing.

| model | pipeline | head | Dice | F2 | **final** | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| `unet_e20` | solution 1 | 3-class | 0.7456 | 0.9391 | **0.8752** | 182 | 19 | 10 |
| `unet_f0` | solution 1 | 3-class | 0.7457 | 0.9362 | **0.8727** | 182 | 22 | 10 |
| `arch_unetpp_r34` | solution 1 | 3-class | 0.7288 | 0.9330 | **0.8500** | 181 | 21 | 11 |
| `arch_unet_r34` | solution 1 | 3-class | 0.7311 | 0.9242 | **0.8446** | 178 | 17 | 14 |
| `arch_unet_effb0` | solution 1 | 3-class | 0.7311 | 0.9232 | **0.8438** | 178 | 18 | 14 |
| `alb_unet_resnet34_f0` | **solution 2** | binary | 0.7143 | 0.9133 | **0.8155** | 177 | 16 | 17 |

Solution 2 lands below all ten solution-1 models on this fold — 0.0597 behind
the best, 0.0283 behind the worst.

## The clean comparison

`arch_unet_r34` is the control: same architecture, same encoder, same ImageNet
weights, same fold. It scores 0.8446 against solution 2's 0.8155.

That isolates **−0.0291 to the pipeline change alone** — albumentations stack,
binary head and BCE+Dice loss together, with architecture held constant. The
remaining gap to the best solution-1 model (0.8752) is the already-measured
cost of a pretrained encoder on this data, not anything new.

## What the error counts say

FN went 10 → 17. Solution 2 misses 70% more failing specimens than the best
solution-1 model, while raising 3 fewer false alarms. Under F2 that is the
wrong trade by a wide margin — a miss is weighted 4x a false alarm, and on a
QA line a missed void is a part that ships.

## Most likely causes, in order

1. **Augmentation coverage.** The Compose has flips, rot90 and brightness/
   contrast. Solution 1 adds scale jitter, JPEG re-encode, sensor noise and
   defocus blur, each added to close a *measured* gap. Scale jitter matters
   even in-domain here: fibre radius varies across micrographs (2–22 px), so
   scale tolerance is not only a Test-set concern. Flips and rot90 are also
   partly redundant — they are already baked into the on-disk "Augmented data
   set" folders.
2. **Binary head.** Predicting void against everything else discards the
   fibre/matrix distinction as auxiliary signal. The 3-class formulation makes
   the network account for what a non-void dark region *is*, which is exactly
   the confusion that produces false positives in resin-rich areas.
3. **Pretrained encoder.** Already established on this data: every ImageNet
   backbone lost to the scratch U-Net. Natural-image priors do not transfer to
   micrographs.

## Cheapest next test

Add the four missing transforms to the Compose (`A.RandomScale` or
`A.ShiftScaleRotate`, `A.ImageCompression`, `A.GaussNoise`, `A.GaussianBlur`)
and retrain — 5 minutes of GPU. If the number moves most of the way back to
0.8446, cause 1 dominates and the pipeline is fine once its augmentation stack
matches. If it barely moves, the binary head is carrying the loss.
