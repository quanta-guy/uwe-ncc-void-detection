"""Inject synthetic voids into real micrographs and measure what gets caught.

    python src/inject.py --ckpt runs/unet_f0.pt ... --n 40

Every other measurement here is scored against ImageJ labels we have shown to
miss real voids. This one is not: we place the defect ourselves, so the ground
truth is exact by construction. What it buys is the number a QA engineer
actually asks for - the smallest defect the system reliably catches.

Voids are drawn only into resin-rich areas, because that is where porosity
forms; painting one over a fibre would be a defect the material cannot have
and would flatter the model with an unnaturally easy target. Appearance is
taken from the measured statistics of real voids: mean RGB (63, 59, 90),
darker and bluer than matrix (177, 155, 161), with a feathered edge so the
model is not simply spotting a hard synthetic cut-out.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import uniform_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data import VOID_CLASS, index_training, load_image  # noqa: E402
from predict import load_nets, to_mask, void_prob  # noqa: E402

VOID_RGB = np.array([63, 59, 90], dtype=np.float32)
# Equivalent diameters spanning the real distribution: p10 5.3, p50 15.2, p90 40.
SIZES = [6, 10, 15, 22, 32, 45]


def resin_mask(img, fibre_quantile=0.55, window=9):
    """Where the image is locally dark-ish, i.e. matrix rather than fibre.

    Fibres are the bright circles; a local mean below the median picks out the
    resin between and around them, which is where porosity actually sits.
    """
    grey = img.mean(axis=2)
    local = uniform_filter(grey, size=window)
    return local < np.quantile(local, fibre_quantile)


def inject(img, diameter, rng, tries=60):
    """Paint one elliptical void of the given equivalent diameter.

    Returns (image, boolean mask) or (None, None) if nowhere suitable was found.
    """
    h, w = img.shape[:2]
    ok = resin_mask(img)
    r = diameter / 2.0

    for _ in range(tries):
        cy = rng.integers(int(r) + 3, h - int(r) - 3)
        cx = rng.integers(int(r) + 3, w - int(r) - 3)
        if not ok[cy, cx]:
            continue

        # Slightly irregular ellipse: real voids are not circles.
        ry = r * rng.uniform(0.7, 1.3)
        rx = r * r / max(ry, 1e-3)
        layer = Image.new("L", (w, h), 0)
        ImageDraw.Draw(layer).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        layer = layer.rotate(rng.uniform(0, 180), center=(cx, cy))
        # Feather the edge so the model cannot key on a hard synthetic cut.
        soft = np.array(layer.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0

        if (soft > 0.5).sum() < 4:
            continue

        alpha = soft[..., None]
        tint = VOID_RGB + rng.normal(0, 6, 3).astype(np.float32)
        out = (img.astype(np.float32) * (1 - alpha) + tint * alpha).clip(0, 255).astype(np.uint8)
        return out, soft > 0.5

    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=40, help="Images per size")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results" / "injection"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = load_nets(args.ckpt, device)
    rng = np.random.default_rng(args.seed)

    # Only void-free images, so anything detected at the injection site is
    # unambiguously our synthetic defect and not a pre-existing one.
    df = index_training(args.fold).query("is_val")
    clean = []
    for _, r in df.iterrows():
        from data import load_mask
        if (load_mask(r["mask"]) == VOID_CLASS).sum() == 0:
            clean.append(r)
        if len(clean) >= args.n:
            break
    print(f"{len(clean)} void-free host images from fold {args.fold}\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'diameter':>9} {'injected':>9} {'detected':>9} {'rate':>7} "
          f"{'median IoU':>11} {'false alarms':>13}")
    results = []
    for d in SIZES:
        hits = tot = spurious = 0
        ious = []
        for i, r in enumerate(clean):
            img = load_image(r["image"])
            dirty, truth = inject(img, d, rng)
            if dirty is None:
                continue
            tot += 1

            p, base = void_prob(nets, dirty, device)
            pred = to_mask(p, base, args.threshold, args.min_size) == VOID_CLASS

            inter = (pred & truth).sum()
            # Detected if the model marks a meaningful part of the defect.
            if inter >= max(2, 0.25 * truth.sum()):
                hits += 1
                ious.append(inter / (pred | truth).sum())
            # Anything found well away from the injection is a false alarm.
            if (pred & ~truth).sum() > 20:
                spurious += 1

            if i == 0:
                Image.fromarray(dirty).save(out_dir / f"d{d:02d}_injected.png")
                vis = dirty.copy()
                vis[truth] = [60, 220, 60]
                vis[pred & ~truth] = [255, 60, 60]
                vis[pred & truth] = [255, 220, 60]
                Image.fromarray(vis).save(out_dir / f"d{d:02d}_overlay.png")

        rate = hits / tot if tot else 0.0
        med = float(np.median(ious)) if ious else 0.0
        print(f"{d:9d} {tot:9d} {hits:9d} {rate:6.1%} {med:11.3f} {spurious:13d}")
        results.append((d, tot, hits, rate, med))

    print(f"\ngreen = injected defect, yellow = correctly found, red = elsewhere")
    print(f"examples: {out_dir}")
    small = [r for r in results if r[3] < 0.5]
    if small:
        print(f"\nreliable detection starts above ~{max(r[0] for r in small)} px equivalent diameter")


if __name__ == "__main__":
    main()
