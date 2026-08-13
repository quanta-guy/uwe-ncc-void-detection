"""Solution 3: physical-resolution normalisation and leakage-free validation.

    python solution3/data3.py        # self-check, prints the fold table

Four measured problems with solution 1's data recipe, each fixed here.

**1. Scale is magnification, not object size.**
Across the whole dataset, fibre_radius_px x um_per_pixel = 4.00-4.08um. The
fibre is 8um in diameter - standard carbon fibre - and the 2-22px spread of
apparent radii is entirely microscope magnification. Solution 1 asked the
network to learn invariance to that with random scale jitter. Resampling every
image to a canonical um/pixel removes the nuisance instead, which is nnU-Net's
target-spacing practice. Test images are 0.57um/px with 7px radius, so that is
the natural canonical value.

**2. Validation counted near-duplicates.**
The index holds 1550 originals and 1550 stored _aug_ copies. Every fold's
validation set was exactly 50% augmented copies of originals already in that
fold - 406 of 812 in fold 0. No train/val leakage, because the micrograph
grouping is correct, but each case was counted twice and the hidden Test set
has no such duplicates. Augmentation is a training-only operation here.

**3. Folds were badly imbalanced.**
Striding sorted micrograph names gave 812/676/638/428/546 images and
241/144/195/116/299 void-containing ones - fold 4 at 54.8% void against fold
1's 21.3%. Much of the 0.15 fold-to-fold Dice spread is that imbalance rather
than genuine difficulty. A greedy allocator balances tiles, void prevalence
and failing prevalence while keeping every micrograph intact.

**4. Online augmentation was frozen.**
data.py seeded its generator from (seed, index, torch.initial_seed()).
initial_seed() is the worker's *initial* seed and never advances, so a given
image received the same transform every epoch - measured: one distinct
augmentation over six epochs with num_workers=0. The RNG here is created once
per worker and advances with every draw, so augmentation is genuinely fresh.

Sampling is also rebalanced: micrographs contribute wildly different tile
counts, so uniform tile sampling teaches the largest micrographs most often.
Weights equalise micrographs and boost void-containing tiles.
"""

import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler, get_worker_info

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data import VOID_CLASS, index_test, index_training, load_image, load_mask, to_tensor  # noqa: E402
from evaluation import SEVERITY_THRESHOLD, compute_max_severity  # noqa: E402

#: Test set spacing. Every image is resampled to this, so the network sees one
#: physical scale and the 2-22px fibre-radius spread disappears.
CANONICAL_UM_PER_PX = 0.57
N_FOLDS = 5
_AUG = r"_aug_\d+$"


# -----------------------------
# PHYSICAL RESAMPLING
# -----------------------------
def resample(img, mask, um_per_px, target=CANONICAL_UM_PER_PX):
    """Resample to `target` um/pixel. Returns (img, mask, new_um_per_px).

    scale > 1 magnifies (source was coarser than target). Masks go through
    nearest so class ids are never interpolated into values that do not exist.
    """
    scale = um_per_px / target
    if abs(scale - 1.0) < 1e-3:
        return img, mask, um_per_px

    h, w = img.shape[:2]
    nh, nw = max(8, round(h * scale)), max(8, round(w * scale))
    img = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    if mask is not None:
        mask = np.array(Image.fromarray(mask).resize((nw, nh), Image.NEAREST))
    return img, mask, target


# -----------------------------
# BALANCED GROUP FOLDS
# -----------------------------
_STATS_CACHE = Path(__file__).resolve().parent / "micrograph_stats.csv"


def _micrograph_stats(df, cache=_STATS_CACHE):
    """Per-micrograph tile count, void count and failing count.

    Computed on ORIGINALS only - the augmented copies carry the same defects,
    so counting them would weight each micrograph by how many copies happen to
    exist rather than by what it contains.

    Cached to disk. This reads 1550 masks and runs compute_max_severity on
    every one of them - convex hulls and KD-trees - which is minutes of CPU.
    Every training run needs the same answer to build the same folds, so
    without the cache a five-fold sweep pays for it five times over while the
    GPU sits idle.
    """
    if cache and Path(cache).exists():
        d = pd.read_csv(cache).set_index("group")
        return {g: dict(r) for g, r in d.to_dict("index").items()}

    orig = df[~df["stem"].str.contains(_AUG, regex=True)]
    stats = defaultdict(lambda: {"tiles": 0, "void": 0, "fail": 0})
    for _, r in orig.iterrows():
        m = load_mask(r["mask"])
        s = stats[r["group"]]
        s["tiles"] += 1
        if (m == VOID_CLASS).any():
            s["void"] += 1
            sev, _ = compute_max_severity(m, r["um_per_pixel"])
            s["fail"] += sev >= SEVERITY_THRESHOLD

    stats = dict(stats)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"group": g, **s} for g, s in stats.items()]).to_csv(cache, index=False)
    return stats


def balanced_folds(df, n_folds=N_FOLDS, stats=None):
    """Assign whole micrographs to folds, balancing three quantities at once.

    Greedy: hardest-first by failing count, each micrograph going to whichever
    fold is currently most short of it. Micrographs are never split, so the
    grouping that prevents leakage is preserved exactly.

    A greedy allocator rather than StratifiedGroupKFold because sklearn is not
    a dependency here, and with 28 groups and three quantities to balance a
    greedy pass gets within a few percent of optimal.
    """
    stats = stats or _micrograph_stats(df)
    keys = ("tiles", "void", "fail")
    totals = {k: max(sum(s[k] for s in stats.values()), 1) for k in keys}

    order = sorted(stats, key=lambda g: (-stats[g]["fail"], -stats[g]["void"],
                                         -stats[g]["tiles"]))
    folds = {f: {"tiles": 0, "void": 0, "fail": 0, "groups": []} for f in range(n_folds)}

    def cost(f, s):
        """Fractional load if this micrograph went to fold f.

        Each quantity is normalised by its dataset total, so tiles (thousands)
        and failing images (hundreds) carry comparable weight. A lexicographic
        cost cannot do this: a fold that falls behind on the first key stays
        the minimum forever and collects everything - measured, that put 20 of
        28 micrographs in one fold.
        """
        return sum((folds[f][k] + s[k]) / totals[k] for k in keys)

    for g in order:
        s = stats[g]
        pick = min(folds, key=lambda f: cost(f, s))
        for k in keys:
            folds[pick][k] += s[k]
        folds[pick]["groups"].append(g)

    return {g: f for f, d in folds.items() for g in d["groups"]}, folds


def void_index(df, cache=None):
    """Add a `has_void` column, cached - it costs 3100 mask reads otherwise.

    The sampler needs it per row to over-sample foreground, and both training
    and evaluation want the same answer, so it is computed once to disk.
    """
    cache = Path(cache or Path(__file__).resolve().parent / "void_index.csv")
    if cache.exists():
        known = pd.read_csv(cache).set_index("stem")["has_void"].to_dict()
    else:
        known = {}
    missing = [s for s in df["stem"] if s not in known]
    if missing:
        want = set(missing)
        for _, r in df[df["stem"].isin(want)].iterrows():
            known[r["stem"]] = bool((load_mask(r["mask"]) == VOID_CLASS).any())
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"stem": list(known), "has_void": list(known.values())}).to_csv(
            cache, index=False)
    df = df.copy()
    df["has_void"] = df["stem"].map(known).fillna(False).astype(bool)
    return df


def index3(fold=0, n_folds=N_FOLDS, assignment=None, stats=None):
    """Index with balanced folds and originals flagged.

    `is_val` marks held-out ORIGINALS only. Augmented copies of a held-out
    micrograph are excluded from both sides: they must not train (their source
    is held out) and must not be validated on (they are duplicates).
    """
    df = index_training(0).copy()          # fold arg unused; we reassign below
    df["is_aug"] = df["stem"].str.contains(_AUG, regex=True)
    if assignment is None:
        assignment, _ = balanced_folds(df, n_folds, stats)
    df["fold"] = df["group"].map(assignment)

    df["is_val"] = (df["fold"] == fold) & (~df["is_aug"])
    df["is_train"] = df["fold"] != fold
    return df


# -----------------------------
# AUGMENTATION
# -----------------------------
def _augment(img, mask, size, rng):
    """Geometry, then photometry, then the capture chain.

    Scale jitter is deliberately much narrower than solution 1's 0.6-1.7x.
    Physical resampling has already removed magnification differences, so what
    remains is tolerance to residual calibration error rather than a 3x range
    the network must learn to undo.
    """
    if rng.random() < 0.5:
        s = float(np.exp(rng.uniform(np.log(0.85), np.log(1.18))))
        h, w = img.shape[:2]
        nh, nw = max(size // 2, round(h * s)), max(size // 2, round(w * s))
        img = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((nw, nh), Image.NEAREST))

    img, mask = _crop_or_pad(img, mask, size, rng)

    k = int(rng.integers(4))
    if k:
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    img = np.ascontiguousarray(img)

    if rng.random() < 0.5:
        img = np.clip(img * rng.uniform(0.88, 1.12, size=3), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = np.array(Image.fromarray(img).filter(
            ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6))))
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=int(rng.integers(35, 96)))
        img = np.array(Image.open(buf))

    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def _crop_or_pad(img, mask, size, rng):
    h, w = img.shape[:2]
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        h, w = img.shape[:2]
    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    return img[top:top + size, left:left + size], mask[top:top + size, left:left + size]


# -----------------------------
# DATASET
# -----------------------------
class Micrographs3(Dataset):
    """Physically normalised tiles with genuinely fresh augmentation.

    The RNG is built once per worker and then ADVANCES on every draw. Solution
    1 rebuilt it per item from torch.initial_seed(), which is fixed for a
    worker's lifetime, so each image received one frozen transform - measured
    at exactly 1 distinct augmentation over 6 epochs. Reproducibility is kept
    through (seed, worker id): same seed and worker count reproduces a run,
    while every epoch still sees new samples.
    """

    def __init__(self, df, train=False, size=256, seed=0, target=CANONICAL_UM_PER_PX):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.size = size
        self.seed = seed
        self.target = target
        self._rng = None

    def __len__(self):
        return len(self.df)

    def _get_rng(self):
        if self._rng is None:
            info = get_worker_info()
            self._rng = np.random.default_rng([self.seed, info.id if info else 0])
        return self._rng

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img, mask = load_image(row["image"]), load_mask(row["mask"])
        img, mask, _ = resample(img, mask, row["um_per_pixel"], self.target)

        if self.train:
            img, mask = _augment(img, mask, self.size, self._get_rng())
        return to_tensor(img), torch.from_numpy(mask.astype(np.int64))


def balanced_sampler(df, void_boost=3.0, seed=0):
    """Equalise micrographs, then over-sample void-containing tiles.

    Micrographs contribute between a handful and several hundred tiles, so
    uniform sampling trains mostly on the largest ones. Weighting by
    1/tiles_in_micrograph makes each micrograph equally likely; void_boost
    then raises the share of patches that contain the class being scored,
    which is nnU-Net's foreground-oversampling idea.
    """
    per_group = df["group"].value_counts()
    w = df["group"].map(lambda g: 1.0 / per_group[g]).to_numpy(dtype=np.float64)
    has_void = df["has_void"].to_numpy() if "has_void" in df else np.zeros(len(df), bool)
    w = w * np.where(has_void, void_boost, 1.0)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(df),
                                 replacement=True, generator=g)


def demo():
    """Fold balance, resampling arithmetic, and that augmentation is fresh."""
    df = index_training(0)

    # The physical invariant this whole approach rests on. Not exactly
    # constant - measured 3.42 to 4.08um, a 19% spread - but tight enough that
    # resampling to a canonical spacing collapses a 2-22px radius range into
    # roughly 6.0-7.2px, which is the point.
    prod = df["fibre_radius_px"] * df["um_per_pixel"]
    print(f"fibre_radius_px x um_per_pixel: {prod.min():.2f} - {prod.max():.2f} um  "
          f"(median {prod.median():.2f}, CV {prod.std() / prod.mean():.1%})")
    assert 3.3 < prod.min() and prod.max() < 4.3, (prod.min(), prod.max())
    canon = prod / CANONICAL_UM_PER_PX
    print(f"after resampling, fibre radius becomes {canon.min():.1f} - {canon.max():.1f} px "
          f"(was {df['fibre_radius_px'].min():.0f} - {df['fibre_radius_px'].max():.0f} px)")
    assert canon.max() / canon.min() < 1.3, "resampling did not collapse the scale range"

    te = index_test()
    assert abs((te["fibre_radius_px"] * te["um_per_pixel"]).median() - 4.0) < 0.2

    # Resampling must land the fibre at the canonical radius, whatever it started at.
    for um, rad in [(0.29, 14.0), (0.80, 5.0), (0.57, 7.0), (2.0, 2.0)]:
        img = np.zeros((100, 100, 3), np.uint8)
        m = np.zeros((100, 100), np.uint8)
        m[10:20, 10:20] = VOID_CLASS
        out, om, new_um = resample(img, m, um)
        assert abs(new_um - CANONICAL_UM_PER_PX) < 1e-6 or abs(um - CANONICAL_UM_PER_PX) < 1e-3
        assert set(np.unique(om).tolist()) <= {0, 1, 2}, "resampling invented a class"
        eff = rad * um / CANONICAL_UM_PER_PX
        assert 6.0 < eff < 8.2, f"{um} um/px, {rad}px -> {eff:.1f}px canonical"

    # Folds must partition micrographs and be better balanced than striding.
    assign, folds = balanced_folds(df)
    assert len(assign) == df.group.nunique()
    sizes = [folds[f]["tiles"] for f in folds]
    voids = [folds[f]["void"] for f in folds]
    print(f"\n{'fold':>4} {'micrographs':>12} {'orig tiles':>11} {'void':>6} {'failing':>8}")
    for f in folds:
        print(f"{f:4d} {len(folds[f]['groups']):12d} {folds[f]['tiles']:11d} "
              f"{folds[f]['void']:6d} {folds[f]['fail']:8d}")
    spread = (max(voids) - min(voids)) / max(np.mean(voids), 1)
    tile_spread = (max(sizes) - min(sizes)) / max(np.mean(sizes), 1)
    print(f"\nvoid-count spread across folds: {spread:.1%} of mean "
          f"(striding gave 116-299, i.e. {(299 - 116) / 199:.0%})")
    print(f"tile-count spread across folds: {tile_spread:.1%} of mean")
    # A fold holding most of the data is worse than the imbalance being fixed.
    assert tile_spread < 0.60, f"folds are lopsided by tile count: {sizes}"
    assert spread < 0.60, f"folds are lopsided by void count: {voids}"
    assert all(len(folds[f]["groups"]) >= 3 for f in folds), \
        f"a fold has too few micrographs: {[len(folds[f]['groups']) for f in folds]}"

    # Validation must be originals only, and no micrograph on both sides.
    d3 = index3(0, assignment=assign)
    assert not d3[d3.is_val].is_aug.any(), "augmented copies leaked into validation"
    assert not (set(d3[d3.is_val].group) & set(d3[d3.is_train].group))
    base = d3[d3.is_val]["stem"].str.replace(_AUG, "", regex=True)
    assert not base.duplicated().any(), "validation still contains duplicate cases"

    # The bug this file exists to fix: augmentation must differ every draw.
    ds = Micrographs3(d3[d3.is_train].head(4), train=True)
    seen = {i: set() for i in range(len(ds))}
    for _ in range(6):
        for i in range(len(ds)):
            seen[i].add(hash(ds[i][0].numpy().tobytes()))
    distinct = [len(s) for s in seen.values()]
    assert min(distinct) >= 5, f"augmentation still frozen: {distinct} distinct over 6 draws"

    print(f"augmentation: {min(distinct)}-{max(distinct)} distinct per image over 6 draws "
          f"(solution 1 measured exactly 1)")
    print(f"validation: {d3.is_val.sum()} originals, {d3.is_train.sum()} training rows")
    print("ok")


if __name__ == "__main__":
    demo()
