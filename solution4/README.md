# Solution 4 — solution 1 with the augmentation RNG unfrozen

One variable. Everything else is imported from solution 1 rather than copied,
so it cannot drift: the split and `_augment` from `src/data.py`, the model from
`src/model.py`, the loss and validation metric from `src/train.py`. The only
difference is `Micrographs4` in place of `Micrographs`.

## The bug

`src/data.py` builds its generator per item:

```python
rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**32))
```

`torch.initial_seed()` returns the worker's **initial** seed. It is fixed for
that worker's lifetime and never advances, so `(seed, i, worker_seed)` is
constant for a given image in a given worker.

Measured, `num_workers=0`, 6 epochs:

```
distinct augmentations per image    solution 1  [1, 1, 1, 1]
                                    solution 4  [6, 6, 5, 6]
```

Byte-identical every epoch. With workers it is only slightly better — shuffling
sends an index to different workers on different epochs, so an image sees at
most `num_workers` transforms, cycling.

So the scale jitter, per-channel colour gain, defocus blur, sensor noise and
JPEG re-encode that solution 1 was designed around — each added to close a
*measured* gap to the Test set — were drawn a handful of times and then frozen
for the entire run.

## The fix

One generator per worker, created on first use and advancing on every draw.
Built lazily rather than in `__init__`, because DataLoader forks after
construction and a generator made in the parent would be copied identically
into every worker — the same bug in another form.

`(seed, worker_id)` keeps runs reproducible: same seed and worker count
reproduces a run exactly, while every epoch still sees fresh samples. Verified
both ways in `data4.py`'s self-check — same seed reproduces, different seeds
diverge.

## Protocol

Scored by the **same non-nested OOF sweep** that produced solution 1's 0.8869,
on the same striding split, over the same threshold grid, including the stored
`_aug_` duplicates in validation.

That protocol is optimistic — it selects threshold and `min_size` on the data it
reports, which solution 3 fixed with nested selection. It is used here anyway
because **both arms are equally optimistic**, so the difference between them
remains a fair read on what the RNG fix bought. Mixing protocols would confound
the one variable under test.

Three seeds, because the measured run-to-run noise floor on this data is
**0.0131** Dice. A single pair of runs cannot resolve anything smaller.

## Run

```bash
python solution4/data4.py          # proves solution 1 frozen, solution 4 fresh
python solution4/train4.py --demo

python solution4/train4.py --epochs 20 --fold 0 --seed 0
python solution4/evaluate4.py --runs runs --pattern "unet_f{f}.pt" --tag solution1_baseline
python solution4/evaluate4.py --runs solution4/runs --pattern "s4_unet_f{f}_s0.pt"
```

`bash l4_run4.sh` does three seeds × five folds, the baseline, and one score
per seed.

## How to read the result

- **Solution 4 clearly ahead** (> 0.0131) — solution 1 has been leaving that on
  the table all competition, and the augmentation work was sound but inert.
- **No difference** — the augmentation was doing little even when working, and
  the effort spent designing it was misplaced. That is a real finding too: it
  would mean the on-disk "Augmented data set" copies already supplied the
  variety, and the online stack was redundant.
- **Solution 4 behind** — unlikely, but it would mean the frozen transforms
  were acting as an accidental regulariser on 22 micrographs.

Whichever way it lands, it is measured on the shipping model rather than on a
variant, so it applies directly.
