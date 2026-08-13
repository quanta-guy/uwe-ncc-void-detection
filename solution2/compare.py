"""Compare models across all three classes, and stitch panels to look at.

    python solution2/compare.py --ckpt runs/unet_f0.pt solution2/runs/alb_unet_f0.pt ...
    python solution2/compare.py                      # every fold-0 model it can find

Two things the leaderboard score cannot tell you.

**All three classes.** The challenge scores void only - dice_void and
compute_max_severity both ignore matrix and fibre entirely. A model can
therefore produce a mask that is excellent where it is graded and nonsense
everywhere else, and no number in results/ would move. This reports per-class
Dice, IoU and a confusion matrix, so the fibre and matrix work is visible.

**Panels.** original | ground truth | model 1 | model 2 | ... in one strip per
image, same colours throughout, worst disagreement first. Numbers say a model
lost; a panel says why, and on this data it has already shown the ground truth
itself to be wrong in both directions.

Predictions here are plain argmax over the three channels - what the model
actually says, before the tuned threshold and despeckling that the scored
pipeline applies to the void class. That is deliberate: the threshold is fitted
per model, so comparing post-threshold masks would compare fitted knobs as much
as models.

Handles both families. Solution 1 checkpoints take raw [0,1] and softmax;
solution 2 takes an albumentations Compose and sigmoid. Which is which is read
off the checkpoint, never guessed.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import N_CLASSES, VOID_CLASS, index_training, load_image, load_mask, to_tensor  # noqa: E402
from model import build  # noqa: E402
from pipeline import build_transforms  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "compare"
CLASS_NAMES = ["matrix", "fibre", "void"]
#: Dark ground, pale fibres, hot void - the void has to pop out of a panel at a
#: glance, and matrix/fibre must stay distinguishable without being loud.
LUT = np.array([[45, 45, 60], [205, 200, 190], [225, 55, 55]], np.uint8)


def load_any(path, device):
    """Rebuild a checkpoint from either solution. Returns (net, meta)."""
    ckpt = torch.load(path, map_location=device)
    alb = ckpt.get("pipeline", "").startswith("albumentations")
    arch = ckpt.get("arch", "unet")

    if ckpt.get("binary"):
        return None, {"skip": "binary head - cannot segment three classes"}

    if arch == "unet":
        net, _ = build(arch, ckpt.get("base", 32), ckpt.get("depth", 4),
                       ckpt.get("chroma", False))
    else:
        net = getattr(smp, arch)(encoder_name=ckpt["encoder"], encoder_weights=None,
                                 in_channels=3, classes=ckpt.get("classes", N_CLASSES))
    net = net.to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    meta = {"name": Path(path).stem, "solution": 2 if alb else 1, "arch": arch,
            "encoder": ckpt.get("encoder", "-"), "weights": ckpt.get("weights", "-"),
            "aug": ckpt.get("aug", "-"), "norm": ckpt.get("norm", "raw[0,1]"),
            "fold": ckpt.get("fold", 0), "val_dice": ckpt.get("val_dice", float("nan"))}
    # Solution 2 normalises in the Compose; solution 1 feeds raw [0,1]. Getting
    # this backwards still produces plausible-looking maps, just worse ones.
    meta["transform"] = build_transforms(ckpt["norm"])[1] if alb else None
    return net, meta


@torch.no_grad()
def predict(net, meta, img, device):
    """(H, W) class map by argmax over the three channels."""
    h, w = img.shape[:2]
    if meta["transform"] is not None:
        x = meta["transform"](image=img)["image"][None].to(device)
        act = torch.sigmoid          # multilabel BCE head
    else:
        x = to_tensor(img)[None].to(device)
        act = lambda t: t.softmax(dim=1)   # noqa: E731  - mutually exclusive head
        ph, pw = (-h) % 16, (-w) % 16
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")

    with torch.autocast(device.type, enabled=device.type == "cuda"):
        p = act(net(x).float())
    return p[0, :, :h, :w].argmax(dim=0).cpu().numpy().astype(np.uint8)


def tally(pred, gt, hits):
    """Accumulate per-class intersection/areas and the confusion matrix."""
    for c in range(N_CLASSES):
        p, g = pred == c, gt == c
        hits["inter"][c] += (p & g).sum()
        hits["pred"][c] += p.sum()
        hits["true"][c] += g.sum()
    # np.bincount over the flattened (true, pred) pair is far quicker than a
    # Python loop over 3x3 on 800 images.
    hits["cm"] += np.bincount(gt.ravel().astype(np.int64) * N_CLASSES + pred.ravel(),
                              minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    return hits


def scores(hits):
    """Dice and IoU per class, computed over the whole split at once.

    Pooled rather than averaged per image on purpose: a per-image mean lets the
    thousands of images with three void pixels dominate the handful carrying a
    real defect, which flatters every model equally and hides the differences
    this is meant to expose.
    """
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        i, p, t = hits["inter"][c], hits["pred"][c], hits["true"][c]
        out[f"dice_{name}"] = round(2 * i / (p + t), 4) if p + t else float("nan")
        out[f"iou_{name}"] = round(i / (p + t - i), 4) if p + t - i else float("nan")
    out["pixel_acc"] = round(hits["cm"].trace() / hits["cm"].sum(), 4)
    out["mean_dice"] = round(float(np.mean([out[f"dice_{n}"] for n in CLASS_NAMES])), 4)
    return out


def colourise(mask):
    return LUT[np.clip(mask, 0, N_CLASSES - 1)]


def strip(img, gt, preds, labels, band=16):
    """One horizontal comparison strip with a labelled header over each tile."""
    tiles = [img, colourise(gt)] + [colourise(p) for p in preds]
    h, w = img.shape[:2]
    gap = 3
    canvas = Image.new("RGB", (len(tiles) * w + (len(tiles) - 1) * gap, h + band), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x = i * (w + gap)
        canvas.paste(Image.fromarray(tile.astype(np.uint8)), (x, band))
        draw.text((x + 2, 3), label[:34], fill=(0, 0, 0))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", help="Checkpoints; default is every fold-0 model found")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--panels", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = args.ckpt or ([str(REPO / "runs" / f"unet_f{args.fold}.pt")]
                          + sorted(str(p) for p in (REPO / "solution2" / "runs").glob("*.pt")))

    nets, metas = [], []
    for p in paths:
        if not Path(p).exists():
            print(f"  skip {Path(p).name}: not found")
            continue
        net, meta = load_any(p, device)
        if net is None:
            print(f"  skip {Path(p).name}: {meta['skip']}")
            continue
        nets.append(net)
        metas.append(meta)
        print(f"  {meta['name']:34} solution {meta['solution']}  {meta['arch']}/"
              f"{meta['encoder']}  aug={meta['aug']}  norm={meta['norm']}")
    if not nets:
        sys.exit("no usable checkpoints")

    folds = {m["fold"] for m in metas}
    if folds != {args.fold}:
        print(f"\n  WARNING: checkpoints span folds {sorted(folds)}; grading all on "
              f"fold {args.fold}, so any model trained on it is marking its own homework")

    df = index_training(args.fold).query("is_val")
    if args.limit:
        df = df.head(args.limit)
    print(f"\nfold {args.fold}: {len(df)} held-out images, {df.group.nunique()} micrographs\n")

    hits = [{"inter": np.zeros(N_CLASSES, np.int64), "pred": np.zeros(N_CLASSES, np.int64),
             "true": np.zeros(N_CLASSES, np.int64),
             "cm": np.zeros((N_CLASSES, N_CLASSES), np.int64)} for _ in nets]
    gallery = []

    for _, row in df.iterrows():
        img = load_image(row["image"])
        gt = load_mask(row["mask"])
        preds = [predict(n, m, img, device) for n, m in zip(nets, metas)]
        for h, p in zip(hits, preds):
            tally(p, gt, h)

        # Rank for the gallery by how much the models disagree with each other
        # on the void class. Images everyone agrees on teach nothing; the ones
        # that split the field are where the remaining error lives.
        voids = [(p == VOID_CLASS) for p in preds]
        union = np.zeros_like(voids[0])
        inter = np.ones_like(voids[0])
        for v in voids:
            union |= v
            inter &= v
        disagree = int(union.sum() - inter.sum())
        if union.any() or (gt == VOID_CLASS).any():
            gallery.append((disagree, row["stem"], img, gt, preds))

    print(f"{'model':34} {'d_matrix':>9} {'d_fibre':>8} {'d_void':>7} "
          f"{'mean':>7} {'px_acc':>7}")
    rows = []
    for meta, h in zip(metas, hits):
        s = scores(h)
        rows.append({"model": meta["name"], "solution": meta["solution"],
                     "arch": meta["arch"], "encoder": meta["encoder"],
                     "aug": meta["aug"], "norm": meta["norm"], **s})
        print(f"{meta['name'][:34]:34} {s['dice_matrix']:9.4f} {s['dice_fibre']:8.4f} "
              f"{s['dice_void']:7.4f} {s['mean_dice']:7.4f} {s['pixel_acc']:7.4f}")

    out_dir = Path(args.out)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_class.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nconfusion (rows = truth, cols = prediction, % of each true class)")
    for meta, h in zip(metas, hits):
        cm = h["cm"] / h["cm"].sum(axis=1, keepdims=True).clip(min=1)
        print(f"  {meta['name'][:40]}")
        for c, name in enumerate(CLASS_NAMES):
            print(f"    {name:7} " + "  ".join(f"{cm[c, j]:6.1%}" for j in range(N_CLASSES)))

    gallery.sort(key=lambda g: -g[0])
    labels = ["original", "ground truth"] + [m["name"][:34] for m in metas]
    for disagree, stem, img, gt, preds in gallery[:args.panels]:
        strip(img, gt, preds, labels).save(out_dir / "panels" / f"{disagree:07d}_{stem}.png")

    print(f"\n{len(gallery[:args.panels])} panels (worst disagreement first) -> {out_dir / 'panels'}")
    print(f"per-class metrics -> {out_dir / 'per_class.csv'}")
    print(f"colours: matrix dark, fibre pale, void red")


if __name__ == "__main__":
    main()
