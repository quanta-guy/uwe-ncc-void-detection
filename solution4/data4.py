"""Solution 4: solution 1 with one line changed - the augmentation RNG.

    python solution4/data4.py     # self-check: proves the RNG now advances

Everything else is deliberately identical to solution 1: the same index, the
same VAL_EVERY=5 striding split, the same `_augment` function imported from
src/data.py rather than copied, the same model, loss, schedule and epochs. The
point is a controlled A/B - any difference in the result is attributable to the
RNG and nothing else.

**The bug.** src/data.py builds its generator per item:

    rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**32))

`torch.initial_seed()` returns the worker's *initial* seed. It is fixed for
that worker's lifetime and does not advance, so `(seed, i, worker_seed)` is
constant for a given image in a given worker. Measured with num_workers=0: an
image receives **exactly 1 distinct augmentation over 6 epochs** - byte
identical every time.

With workers the picture is only slightly better. Shuffling sends an index to
different workers on different epochs, so an image sees at most num_workers
distinct transforms, cycling. Either way the scale jitter, colour gain, blur,
noise and JPEG re-encode that solution 1 was built around are drawn a handful
of times and then frozen for the whole run.

**The fix.** One generator per worker, created on first use and advancing on
every draw. Reproducibility is kept through (seed, worker_id): the same seed
and worker count reproduce a run exactly, while every epoch still sees fresh
samples.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

# Imported, not copied. If solution 1's augmentation changes, solution 4 must
# change with it or the comparison stops being about the RNG.
from data import N_CHANNELS, _augment, load_image, load_mask, to_tensor  # noqa: E402


class Micrographs4(Dataset):
    """Identical to src.data.Micrographs except the RNG advances."""

    def __init__(self, df, train=False, size=256, seed=0):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.size = size
        self.seed = seed
        self._rng = None

    def __len__(self):
        return len(self.df)

    def _get_rng(self):
        """One generator per worker process, built once, then advanced.

        Built lazily rather than in __init__ because DataLoader forks after
        construction - a generator made in the parent would be duplicated
        identically into every worker, which is a different version of the same
        bug.
        """
        if self._rng is None:
            info = get_worker_info()
            self._rng = np.random.default_rng([self.seed, info.id if info else 0])
        return self._rng

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_image(row["image"])
        mask = load_mask(row["mask"])
        if self.train:
            img, mask = _augment(img, mask, self.size, self._get_rng())
        return to_tensor(img), torch.from_numpy(mask.astype(np.int64))


def demo():
    """The whole point: solution 1 freezes, solution 4 does not."""
    sys.path.insert(0, str(REPO / "src"))
    from data import Micrographs, index_training

    df = index_training(0)
    sub = df[~df.is_val].head(4)

    def distinct(ds, draws=6):
        seen = {i: set() for i in range(len(ds))}
        for _ in range(draws):
            for i in range(len(ds)):
                seen[i].add(hash(ds[i][0].numpy().tobytes()))
        return [len(s) for s in seen.values()]

    old = distinct(Micrographs(sub, train=True))
    new = distinct(Micrographs4(sub, train=True))
    print(f"distinct augmentations per image over 6 draws (num_workers=0)")
    print(f"  solution 1  {old}")
    print(f"  solution 4  {new}")
    assert max(old) == 1, f"expected solution 1 to be frozen, got {old}"
    assert min(new) >= 5, f"solution 4 still frozen: {new}"

    # Shapes and ranges must be untouched - this is meant to be the only change.
    x, y = Micrographs4(sub, train=True)[0]
    assert x.shape == (N_CHANNELS, 256, 256) and y.shape == (256, 256)
    assert x.dtype == torch.float32 and 0 <= x.min() and x.max() <= 1
    assert y.max() <= 2

    # Same seed and worker id must reproduce the same stream, or runs are not
    # repeatable and every comparison becomes noise.
    a = [hash(Micrographs4(sub, train=True, seed=7)[i][0].numpy().tobytes()) for i in range(4)]
    b = [hash(Micrographs4(sub, train=True, seed=7)[i][0].numpy().tobytes()) for i in range(4)]
    assert a == b, "same seed did not reproduce the same augmentations"

    c = [hash(Micrographs4(sub, train=True, seed=8)[i][0].numpy().tobytes()) for i in range(4)]
    assert a != c, "different seeds produced identical augmentations"

    print("ok  reproducible per (seed, worker), fresh every epoch")


if __name__ == "__main__":
    demo()
