"""Dataset indexing and loading for the CFRP void-detection challenge.

Two things here are not cosmetic:

1. The train/val split groups by source *micrograph*, not by file. 4000 rows
   are only 3100 distinct files (Data set II re-uses 250 images from Data set
   I), which are only 1550 originals (the rest are `_aug_N` transforms), which
   are only 28 micrographs (the rest are tiles cut from them). Splitting any
   finer puts a near-copy - or a directly adjacent tile of the same slide - of
   every val image into train, and the local score stops meaning anything.
   The Test set is 2 entirely unseen micrographs, so micrograph-level holdout
   is the split that resembles judging.

2. The augmentations added here each cover a measured or structural gap to
   the Test set - JPEG re-encoding (Test is all JPEG, training is nearly all
   TIFF), scale jitter (below), a colour cast (Test images are noticeably
   more saturated), and defocus/noise (the on-disk blur is applied once and
   frozen; nothing on disk adds noise at all). Flips, crops, brightness,
   contrast and CLAHE are already baked into the "Augmented data set" folders
   on disk, so they are not redone here.

   On scale: the original data has 9 fibre radii (2-6, 10, 11, 14, 22 px) and
   the Test set's 7 px is not among them. The on-disk augmentation reaches 7
   only as a side effect of its random resized crop, and that crop only ever
   zooms *in* - one fixed sample per original. The jitter here spans 0.6-1.7x
   fresh every epoch, so it covers zooming out too.

Images are kept in colour rather than converted to grey: averaged over Data
set I, void pixels read (63, 59, 90) against matrix (177, 155, 161), so voids
are not merely darker, they are bluer. That is signal a grey channel throws
away.
"""

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "Data sets"

TRAIN_SETS = ["Data set I", "Data set II", "Data set III"]
SPLITS = ["Original data set", "Augmented data set"]
TEST_SET = "Test data set"

VOID_CLASS = 2
N_CLASSES = 3
N_CHANNELS = 3
VAL_EVERY = 5  # every 5th micrograph is held out -> ~20% val

_AUG_SUFFIX = re.compile(r"_aug_\d+$")
_TILE_SUFFIX = re.compile(r"_\d+$")


def micrograph_of(stem):
    """Source micrograph a file came from.

    `2_3_1_R_cut_128_1024_aug_0` -> `2_3_1_R_cut`, dropping the augmentation
    index, the tile offset and the cut size in that order.
    """
    stem = _AUG_SUFFIX.sub("", stem)
    return _TILE_SUFFIX.sub("", _TILE_SUFFIX.sub("", stem))


# -----------------------------
# INDEX
# -----------------------------
def index_training(fold=0):
    """One row per image/mask pair across Data sets I, II and III.

    `fold` selects which slice of micrographs is held out, 0 to VAL_EVERY-1.
    The folds partition the 28 micrographs exactly, so training one model per
    fold gives every micrograph a model that never saw it - which is what
    makes an honest out-of-fold estimate possible, and gives the ensemble
    genuinely different models rather than reruns of one split.
    """
    rows = []
    for ds in TRAIN_SETS:
        for split in SPLITS:
            folder = DATA / ds / split
            meta = pd.read_csv(folder / "metadata.csv")
            for _, r in meta.iterrows():
                stem = Path(r["image_id"]).stem
                rows.append({
                    "stem": stem,
                    "image": str(folder / "Images" / r["image_id"]),
                    "mask": str(folder / "Masks" / f"{stem}.png"),
                    "um_per_pixel": float(r["um_per_pixel"]),
                    "fibre_radius_px": float(r["fibre_radius_px"]),
                    "group": micrograph_of(stem),
                })

    df = pd.DataFrame(rows).drop_duplicates("stem").reset_index(drop=True)

    # Only 28 groups exist, so a hash split would be lumpy by luck. Striding
    # sorted names is deterministic and spreads val across the materials,
    # whose names sort into distinct prefixes.
    val_groups = set(sorted(df["group"].unique())[fold::VAL_EVERY])
    df["is_val"] = df["group"].isin(val_groups)
    return df


def index_test():
    """One row per Test image. No masks - that is what we predict."""
    folder = DATA / TEST_SET
    meta = pd.read_csv(folder / "metadata.csv")
    return pd.DataFrame([{
        "stem": Path(r["image_id"]).stem,
        "image": str(folder / "Images" / r["image_id"]),
        "mask": None,
        "um_per_pixel": float(r["um_per_pixel"]),
        "fibre_radius_px": float(r["fibre_radius_px"]),
    } for _, r in meta.iterrows()])


# -----------------------------
# IO
# -----------------------------
def load_image(path):
    """HxWx3 uint8 RGB."""
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.uint8)


# -----------------------------
# AUGMENTATION
# -----------------------------
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


def _augment(img, mask, size, rng):
    # Scale jitter: the Test set's 7 px fibre radius sits between the radii we
    # train on, so the model has to be scale-tolerant rather than memorise one
    # fibre size.
    if rng.random() < 0.7:
        s = float(np.exp(rng.uniform(np.log(0.6), np.log(1.7))))
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

    # Colour cast: the Test images are more saturated than the training ones,
    # and the void class is identified partly by its blue tint, so the model
    # must not anchor on one white balance.
    if rng.random() < 0.5:
        gain = rng.uniform(0.88, 1.12, size=3)
        img = np.clip(img * gain, 0, 255).astype(np.uint8)

    # Defocus then sensor noise then compression - the order a real capture
    # applies them, so the artefacts compose the way they do in the Test set.
    # The on-disk augmentation blurs once and freezes it; these are redrawn
    # every epoch. Kept at p=0.3 so clean images still dominate: degradation
    # robustness is bought with a little clean-image accuracy, and the Dice
    # gate is not yet saturated.
    if rng.random() < 0.3:
        img = np.array(Image.fromarray(img).filter(
            ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6))))

    if rng.random() < 0.3:
        noise = rng.normal(0, rng.uniform(2, 10), img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # JPEG re-encode: every Test image is JPEG, most training images are TIFF.
    if rng.random() < 0.5:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=int(rng.integers(35, 96)))
        img = np.array(Image.open(buf))

    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


# -----------------------------
# DATASET
# -----------------------------
class Micrographs(Dataset):
    def __init__(self, df, train=False, size=256, seed=0):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.size = size
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_image(row["image"])
        mask = load_mask(row["mask"])

        if self.train:
            # Seeded per (epoch-agnostic) call so workers do not all draw the
            # same stream, while a fixed seed still reproduces a run.
            rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**32))
            img, mask = _augment(img, mask, self.size, rng)

        return to_tensor(img), torch.from_numpy(mask.astype(np.int64))


def to_tensor(img):
    """HxWx3 uint8 -> 3xHxW float in [0, 1]. Used by training and inference."""
    return torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)


def demo():
    """Smallest check that the index and the split behave."""
    assert micrograph_of("2_3_1_R_cut_128_1024_aug_0") == "2_3_1_R_cut"
    assert micrograph_of("0_ISC_400_R2_500x_to_1200_10240") == "0_ISC_400_R2_500x_to"

    df = index_training()
    assert len(df) == 3100, len(df)  # 4000 rows, 900 of them shared between sets
    assert df["stem"].is_unique
    # No micrograph may appear on both sides of the split.
    assert not (set(df[df.is_val].group) & set(df[~df.is_val].group))
    assert 0.10 < df["is_val"].mean() < 0.35, df["is_val"].mean()

    # Several draws, so both the upscale and the downscale-then-pad branch of
    # the scale jitter get exercised.
    ds = Micrographs(df[~df.is_val].head(12), train=True)
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (N_CHANNELS, 256, 256) and y.shape == (256, 256), (x.shape, y.shape)
        assert x.dtype == torch.float32 and 0 <= x.min() and x.max() <= 1
        assert y.max() <= 2

    # The folds must partition the micrographs: every one held out exactly
    # once, or an out-of-fold estimate silently double-counts or skips images.
    seen = [set(index_training(f).query("is_val").group) for f in range(VAL_EVERY)]
    assert sum(len(s) for s in seen) == df.group.nunique()
    assert set().union(*seen) == set(df.group.unique())

    test = index_test()
    assert len(test) == 32, len(test)
    print(f"ok  train={(~df.is_val).sum()}  val={df.is_val.sum()}  test={len(test)}  "
          f"micrographs={df.group.nunique()} ({df[df.is_val].group.nunique()} held out)")


if __name__ == "__main__":
    demo()
