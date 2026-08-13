# Solution 2 vs solution 1 — measured

All numbers from `evaluation.py` via `predict._sweep`, so both attempts are
scored by identical code. `final = F2 × min(1, mean_Dice_void / 0.8)`.

## The noise floor comes first

The same configuration was trained twice, once on an L4 and once on a laptop,
changing nothing but the machine and the seed:

| config | laptop | L4 | delta |
|---|---|---|---|
| full aug, scratch U-Net | 0.7148 | 0.7115 | +0.0033 |
| thin aug, scratch U-Net | 0.7087 | 0.7218 | −0.0131 |

**Run-to-run spread on one fold: 0.0131 val Dice.** Nothing smaller than that
is a result. This single measurement invalidated two comparisons we had already
run and reframed a third, so it is the first thing on the page.

For scale: fold-to-fold spread is 0.1525 for solution 1 and 0.2138 for solution
2 — ten to sixteen times the noise floor. **Which micrographs you hold out
matters far more than anything about the model.**

## Out-of-fold, all 28 micrographs, 3100 images

The only honest comparison. Every image scored by the one model that never
trained on it.

| | threshold | Dice | F2 | final | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| solution 1 | 0.4 | 0.7562 | 0.9383 | **0.8869** | 769 | 105 | 37 |
| solution 2 MicroNet 45ep | 0.2 | 0.7455 | **0.9398** | **0.8758** | 768 | 94 | 38 |
| solution 2 `alb_unet` | 0.12 | 0.7394 | 0.9363 | **0.8654** | 762 | 83 | 44 |
| solution 2 `alb_unet` | 0.02 | 0.7291 | **0.9484** | 0.8643 | 779 | 104 | **27** |

Solution 1 to MicroNet: **0.0111 - below the noise floor.** They are tied. MicroNet
even takes F2 (0.9398 against 0.9383) with 11 fewer false alarms and the same
detection, losing only through the Dice gate.

MicroNet's gain over `alb_unet` is +0.0104 out-of-fold, matching its +0.0102
mean across the five folds. Consistent, and consistently inside noise.

**No fold-based measurement separates these models.** Only the Test set does.

## The Test set breaks the tie

32 images, two micrographs never seen in training, all JPEG, fibre radius 7px -
not among the nine radii in the training data. There are no ground-truth masks
here, so this measures behaviour, not accuracy.

| total predicted void area | 32 images | `17-5-2_zoom_mid` | `2-6-1_mid` |
|---|---|---|---|
| `unet_f0` (solution 1, scratch) | 2844 um2 | 2 tiles / 627 | 5 tiles / 2216 |
| `alb_unet_f0` (solution 2, scratch) | 1960 um2 | 2 tiles / 405 | 4 tiles / 1556 |
| MicroNet 45ep (resnet50) | 1443 um2 | **0 tiles / 0** | 4 tiles / 1443 |

**MicroNet detects nothing across all 16 tiles of `17-5-2_zoom_mid`.** Both
scratch models do. On `17-5-2_zoom_mid_512_512` there is a plainly visible dark
void that `unet_f0` marks at severity 33.7 (FAIL) and MicroNet scores 0.0
(PASS).

Two comparisons separate the causes:

- **`alb_unet` vs MicroNet** (same augmentation, different architecture):
  1960 against 1443, and 405 against 0 on the unseen micrograph. Architecture
  and capacity drive the collapse - 32.5M pretrained parameters fit the 28
  training micrographs and fail off them.
- **`unet_f0` vs `alb_unet`** (same architecture, different augmentation):
  2844 against 1960. Solution 1's hand-written stack generalises better than
  the albumentations one. It was built to close *measured* Test gaps - JPEG
  re-encode, scale jitter for that 7px radius, colour saturation.

Without labels, more predicted area is not automatically better. Two things
argue it is here: on held-out data solution 1's region area is 1.07x truth, so
it is calibrated rather than over-claiming; and the missed void is visible in
the image.

**This is why solution 1 is the submission.** Not because it won - on folds it
is tied - but because it is the only model still detecting on micrographs it
has never seen. A model measurably equal on all five held-out folds can be the
one that fails on an unseen specimen. With 28 micrographs, fold validation
measures the wrong thing.

Read fold 0 alone and the gap looks like 0.0523 — overstated 2.4×. That is the
fold spread doing exactly what it does, and it is why no single-fold result
appears in this table.

## The Dice gate decides the winner

At threshold 0.02 solution 2 posts a **higher F2 than solution 1** — 0.9484
against 0.9383 — with false alarms effectively tied. It still loses the final
score, entirely because of the gate: Dice 0.7291/0.8 scales it by 0.911, while
solution 1's 0.7562/0.8 gets 0.945. Strip the gate and solution 2 wins.

So the scoring function and the QA line rank these two models differently, and
the difference is measurable rather than rhetorical.

### How far to push that claim

Direct measurement over all 806 failing specimens (`confirm.py`), each image
predicted by the model that never trained on it:

| | caught | missed |
|---|---|---|
| solution 1 @ 0.4 | 773 | 33 |
| solution 2 @ 0.02 | 779 | 27 |

Solution 2 catches 14 that solution 1 misses; solution 1 catches 8 that
solution 2 misses; **19 are missed by both**. Net +6 to solution 2.

That +6 should not be sold as ten fewer defective parts. Many training images
exist as an original and an `_aug_` copy carrying identical ground-truth
severity, and the verdict flips between those copies for **10 specimens under
solution 1 and 8 under solution 2** — 2.6% and 2.1%. The model-specific
disagreements total 22; the self-disagreements total 18. The margin sits inside
the band where a model contradicts itself on near-identical input.

**19 missed by both is the larger number, and the more useful one.** It exceeds
the gap between the two approaches, so that is where detection work belongs.

## Cluster fragmentation

Severity groups voids within 40 µm by single linkage and sums their lengths, so
a chain of small separate voids becomes one high-severity group. Miss a few
links and the cluster splits, and severity does not degrade gracefully - it
collapses. `2_6_3_R_cut_128_12544` has true severity 99.0; solution 1 recovers
enough void pixels to look reasonable and still reports 18.2, a pass.

This is a tail effect, not the norm. Predicted severity as a fraction of truth
across all failing specimens is median **1.03** for solution 1 and **1.06** for
solution 2 (p10 0.88, p90 ~1.5) - both measure severity well on the typical
specimen. But it is the mechanism behind the worst misses, and it means
per-void segmentation error amplifies non-linearly into severity error whenever
defects are clustered.

## A comparison that was never like-for-like

`results/l4-2026-08-13/` was produced with `bench.py --limit 300`. Its
`run.json` records `val_images_per_model: 300`, and every `best.txt` in it says
"graded on fold 0 (300 held-out images)" - 300 of 812, taken as `df.head(300)`,
so the first rows rather than a random sample.

Comparisons **inside** that file remain valid: all 14 models saw the same 300
images, so the 0.0245 architecture spread stands as a relative result. Any
absolute number from it compared against a full-split number does not.

Rescoring on all 812 images through one code path closes most of the apparent
gap:

| fold 0, 812 images, identical code | threshold | Dice | F2 | final | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| solution 1 `unet_f0` | 0.6 | 0.7461 | 0.9225 | **0.8603** | 181 | 24 | 13 |
| solution 2 MicroNet 45ep | 0.1 | 0.7254 | 0.9369 | **0.8495** | 184 | 22 | 10 |

**0.0108 apart - inside the noise floor.** The same pair compared across code
versions looked like 0.0232.

MicroNet is ahead on both error types at once: 184 caught against 181, 10
missed against 13, and *fewer* false alarms (22 against 24). It loses the final
score only through the Dice gate. Better detector, worse boundaries.

## Training length was the only MicroNet lever that worked

| | val Dice | scored (fold 0, 812) |
|---|---|---|
| 20 epochs, shared LR | 0.7137 | 0.8199 |
| 20 epochs, encoder lr ÷10 | 0.7146 | — |
| 45 epochs, encoder lr ÷10 | **0.7313** | **0.8495** |

+0.0296 on the scored metric from the longer schedule. The discriminative
learning rate contributed +0.0009 - nothing. The original 20-epoch run peaked
on its final epoch, which was the signal, and it was nearly buried by pairing
it with an LR change that did not matter.

## All three classes

The score reads void only, so a model could be excellent where graded and
nonsense everywhere else without any number moving. It is not. Pooled per-class
Dice on fold 0's 812 held-out images:

| model | matrix | fibre | void | mean | px acc |
|---|---|---|---|---|---|
| `unet_f0` (solution 1) | 0.8658 | 0.8676 | 0.8176 | 0.8503 | 0.8664 |
| `alb_unet_f0` (solution 2) | 0.8765 | 0.8762 | 0.8389 | 0.8639 | 0.8761 |
| `micronet resnet50` | 0.8657 | 0.8642 | 0.8390 | 0.8563 | 0.8648 |
| `alb_unet_thin_f0` | 0.8972 | 0.9013 | 0.8431 | 0.8805 | 0.8990 |

Solution 2 wins every pooled class including void, while losing the score. Not
a contradiction — pooled Dice weights by pixel area, the scorer averages per
image. **Solution 2 is better on large voids, worse on small ones**, and small
voids decide the per-image mean.

Void recall confirms it: solution 1 **83.5%**, solution 2 full 81.9%, MicroNet
79.0%, thin 78.2%. Solution 1 is the most sensitive model at argmax.

**Void errors go to matrix, essentially never to fibre** — 13.6–18.5% of void
pixels called matrix against 2.3–3.3% called fibre, consistent across all four
models. That is the resin-pool boundary, quantified.

## What did not work

| change | result | verdict |
|---|---|---|
| full vs thin augmentation | 0.0050 | **unresolvable** — under the noise floor, and the sign flips between machines |
| MicroNet pretraining | +0.0045 vs scratch | no measurable benefit |
| MicroNet + discriminative LR (encoder ÷10) | +0.0009 | no measurable benefit |
| Hough circles + KNN refinement | −0.0032 to −0.0151 | **hurts** — does not ship |
| binary vs 3-class head | ~0.000 | no measurable difference |

Three separate explanations for MicroNet's flatness were proposed and none
survived. Domain-specific pretraining on 22 micrographs did not help, and the
"ImageNet failed because of domain gap" hypothesis gets no support from a
microscopy-pretrained encoder failing the same way.

Refinement was scored with and without on held-out data and lost both times.
The likely cause is self-inflicted: `refine()` overwrites the uncertain band
with hard 0.99/0.01, destroying the continuous probability the threshold sweep
depends on.

## What is still open

**The loss.** Solution 1 uses softmax cross-entropy — classes compete, mutual
exclusivity enforced. Solution 2 uses multilabel BCE — channels independent,
nothing punishes predicting void and matrix on the same pixel. It is the one
variable common to every solution-2 run and the only untested explanation left
for the 0.0215 gap.

## Panels

`results/compare/panels/`, sorted worst-disagreement-first, laid out
`original | ground truth | model 1 | model 2 | ...`. Matrix dark, fibre pale,
void red.

On the worst disagreement (`Middle_R_2304_6144_aug_0`) there is a plainly
visible void: solution 1 finds it and the edge strip, `alb_unet_f0` catches a
fragment, `alb_unet_thin_f0` gets a decent chunk, and **MicroNet marks nothing
at all**.
