"""Score a solution-2 checkpoint with the challenge's own scoring code.

    python solution2/evaluate.py --ckpt solution2/runs/alb_unet_f0.pt
    python solution2/evaluate.py --ckpt ... --refine     # + circles and KNN

Scoring goes through predict._sweep, which calls evaluation.py's severity, F2
and Dice directly - so the number printed here is the same number, computed the
same way, as every model in results/. That is the only reason a second attempt
with a different loss and augmentation stack can be compared to the first.

--refine additionally scores the geometric post-process in refine.py, and both
numbers are printed. Refinement is not assumed to help: it either beats the
plain sweep on held-out data or it does not ship.

Preprocessing is rebuilt from the checkpoint's own `norm` field rather than
from a flag. Evaluating under a different normalisation than training used is
the classic silent failure here - the model still runs, still outputs plausible
maps, and just scores worse for no visible reason.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import VOID_CLASS, index_training, load_image, load_mask  # noqa: E402
from model import build  # noqa: E402
from pipeline import build_transforms  # noqa: E402
from predict import _sweep  # noqa: E402
from refine import refine  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def load_net(path, device):
    ckpt = torch.load(path, map_location=device)
    net, _ = build(ckpt.get("arch", "unet"), ckpt.get("base", 32),
                   ckpt.get("depth", 4), ckpt.get("chroma", False))
    net = net.to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"  {Path(path).name}  norm={ckpt['norm']}  fold={ckpt['fold']}  "
          f"val_dice {ckpt.get('val_dice', float('nan')):.4f}")
    return net, ckpt


@torch.no_grad()
def class_prob(nets, transform, img, device):
    """(3, H, W) per-class probability at the image's NATIVE resolution.

    The network sees a resize to 256; the maps are sent back to the original
    grid before anything is thresholded, because severity is measured in
    microns through um_per_pixel and a mask on the wrong grid reports the wrong
    severity for a void it located perfectly.

    Sigmoid, not softmax: the head is trained with multilabel BCE, so each
    channel is its own probability. Applying softmax here would be a different
    function from the one the loss shaped.
    """
    h, w = img.shape[:2]
    x = transform(image=img)["image"][None].to(device)

    with torch.autocast(device.type, enabled=device.type == "cuda"):
        p = torch.stack([n(x).float().sigmoid() for n in nets]).mean(0)
    p = p[0].cpu().numpy()

    if p.shape[1:] != (h, w):
        p = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in p])
    return p


def collect(nets, val_t, df, device, do_refine):
    probs, bases, gts, ums, greys = [], [], [], [], []
    for _, row in df.iterrows():
        img = load_image(row["image"])
        p = class_prob(nets, val_t, img, device)

        if do_refine:
            p_void, base = refine(p, img, row.get("fibre_radius_px", 0))
        else:
            p_void = p[VOID_CLASS]
            base = (p[1] > p[0]).astype(np.uint8)

        probs.append(p_void.astype(np.float16))
        bases.append(base)
        gts.append(load_mask(row["mask"]))
        ums.append(row["um_per_pixel"])
        greys.append(img.mean(axis=2).astype(np.uint8))
    return probs, bases, gts, ums, greys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="One checkpoint, or several to average as an ensemble")
    ap.add_argument("--fold", type=int, default=None,
                    help="Override the held-out fold; defaults to the checkpoint's own")
    ap.add_argument("--refine", action="store_true",
                    help="Also score with Hough circles + KNN post-processing")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--min-sizes", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets, ckpts = [], []
    for c in args.ckpt:
        n, k = load_net(c, device)
        nets.append(n)
        ckpts.append(k)

    norms = {k["norm"] for k in ckpts}
    assert len(norms) == 1, f"cannot ensemble across normalisations: {norms}"
    _, val_t = build_transforms(norms.pop())

    # A model must be graded on the fold it held out, or it is marking its own
    # homework - every training micrograph would be in the val set.
    fold = args.fold if args.fold is not None else ckpts[0]["fold"]
    df = index_training(fold).query("is_val")
    if args.limit:
        df = df.head(args.limit)
    print(f"\ngrading on fold {fold}: {len(df)} held-out images "
          f"({df.group.nunique()} micrographs)")

    name = Path(args.ckpt[0]).stem if len(args.ckpt) == 1 else "ensemble"
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    modes = [("plain", False)] + ([("refined", True)] if args.refine else [])
    for label, do_refine in modes:
        print(f"\n--- {label} ---")
        cached = collect(nets, val_t, df, device, do_refine)
        best, rows = _sweep(*cached, args.thresholds, args.min_sizes, dilations=[0])
        top = max(rows, key=lambda r: r["final"])
        results[label] = (best, top)
        with open(out_dir / f"sweep_{label}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    lines = [f"model            {name}",
             f"checkpoints      {', '.join(Path(c).name for c in args.ckpt)}",
             f"pipeline         albumentations-v2, 3-class, BCEWithLogits + Dice",
             f"normalisation    {ckpts[0]['norm']}",
             f"graded on fold   {fold}  ({len(df)} held-out images)", ""]
    for label, (best, top) in results.items():
        lines += [f"[{label}]",
                  f"  best setting   --threshold {best[1]} --min-size {best[2]}",
                  f"  Dice_void      {top['dice']}",
                  f"  F2             {top['f2']}",
                  f"  final score    {best[0]:.4f}",
                  f"  TP/FP/FN       {top['tp']}/{top['fp']}/{top['fn']}", ""]
    (out_dir / "best.txt").write_text("\n".join(lines))

    print(f"\n{'=' * 62}")
    for label, (best, top) in results.items():
        print(f"solution 2 [{label:8}]  final {best[0]:.4f}   "
              f"Dice {top['dice']:.4f}   F2 {top['f2']:.4f}   "
              f"TP/FP/FN {top['tp']}/{top['fp']}/{top['fn']}")
    if len(results) == 2:
        d = results["refined"][0][0] - results["plain"][0][0]
        print(f"refinement: {d:+.4f}  ({'keep it' if d > 0 else 'drop it'})")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
