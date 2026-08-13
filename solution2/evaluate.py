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
import segmentation_models_pytorch as smp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (VAL_EVERY, VOID_CLASS, index_training, load_image,  # noqa: E402
                  load_mask, to_tensor)
from model import build  # noqa: E402
from pipeline import build_transforms  # noqa: E402
from predict import _sweep  # noqa: E402
from refine import refine  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def load_net(path, device):
    """Rebuild the exact model the checkpoint was saved from.

    Pretrained encoders are reconstructed with encoder_weights=None - the
    trained weights come from the checkpoint, so re-downloading MicroNet here
    would only overwrite them with the starting point.
    """
    ckpt = torch.load(path, map_location=device)
    arch = ckpt.get("arch", "unet")
    if arch == "unet":
        net, _ = build(arch, ckpt.get("base", 32), ckpt.get("depth", 4),
                       ckpt.get("chroma", False))
    else:
        net = getattr(smp, arch)(encoder_name=ckpt["encoder"], encoder_weights=None,
                                 in_channels=3, classes=ckpt.get("classes", 3))
    net = net.to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    # Solution 1 checkpoints carry no `norm`: they were trained on raw [0,1]
    # with a softmax head. Scoring both families through this one path is the
    # only way the two attempts compare on identical code - anything else
    # compares code versions as much as models.
    ckpt.setdefault("norm", None)
    ckpt.setdefault("fold", 0)
    tag = arch if arch == "unet" else f"{arch}/{ckpt['encoder']}<-{ckpt.get('weights')}"
    print(f"  {Path(path).name}  {tag}  "
          f"norm={ckpt['norm'] or 'raw[0,1] (solution 1)'}  fold={ckpt['fold']}  "
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
    if transform is not None:
        x = transform(image=img)["image"][None].to(device)
        act = torch.sigmoid                        # multilabel BCE head
    else:
        # Solution 1: raw [0,1] and a softmax head, reflect-padded to a
        # multiple of 16 because its U-Net halves resolution four times.
        x = to_tensor(img)[None].to(device)
        act = lambda t: t.softmax(dim=1)           # noqa: E731
        ph, pw = (-h) % 16, (-w) % 16
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")

    with torch.autocast(device.type, enabled=device.type == "cuda"):
        p = torch.stack([act(n(x).float()) for n in nets]).mean(0)
    p = p[0, :, :h, :w].cpu().numpy()

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


def run_oof(args, device):
    """Every image scored by the one model that never trained on it.

    A single fold rests on 6 micrographs, and the fold-to-fold spread here is
    0.21 in val Dice - far larger than any difference between models. Scoring
    all 28 micrographs once each is the only comparison that survives that,
    and it is what solution 1's 0.8869 was measured with, so it is also the
    only number the two attempts can be compared on.
    """
    runs = Path(args.runs)
    cached, folds = ([], [], [], [], []), []
    for f in range(VAL_EVERY):
        ckpt_path = runs / args.pattern.format(f=f)
        if not ckpt_path.exists():
            sys.exit(f"missing {ckpt_path} - --oof needs all five folds")
        net, ckpt = load_net(ckpt_path, device)
        assert ckpt["fold"] == f, f"{ckpt_path.name} says fold {ckpt['fold']}, expected {f}"
        val_t = build_transforms(ckpt["norm"])[1] if ckpt["norm"] else None

        df = index_training(f).query("is_val")
        if args.limit:
            df = df.head(args.limit)
        print(f"    fold {f}: {len(df)} held-out images, {df.group.nunique()} micrographs")
        part = collect([net], val_t, df, device, args.refine)
        for dst, src in zip(cached, part):
            dst.extend(src)
        folds.append(ckpt.get("val_dice", float("nan")))

    print(f"\nout-of-fold over {len(cached[0])} images, all 28 micrographs")
    print(f"per-fold val_dice at training time: "
          f"{', '.join(f'{d:.4f}' for d in folds)}  (spread {max(folds) - min(folds):.4f})")

    best, rows = _sweep(*cached, args.thresholds, args.min_sizes, dilations=[0])
    top = max(rows, key=lambda r: r["final"])

    out_dir = Path(args.out) / "oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "best.txt").write_text(
        f"solution 2 out-of-fold, all 28 micrographs, {len(cached[0])} images\n"
        f"per-fold val_dice   {', '.join(f'{d:.4f}' for d in folds)}\n"
        f"best setting        --threshold {best[1]} --min-size {best[2]}\n"
        f"Dice_void           {top['dice']}\nF2                  {top['f2']}\n"
        f"final score         {best[0]:.4f}\nTP/FP/FN            "
        f"{top['tp']}/{top['fp']}/{top['fn']}\n")

    print(f"\n{'=' * 62}")
    print(f"solution 2 OOF  final {best[0]:.4f}   Dice {top['dice']:.4f}   "
          f"F2 {top['f2']:.4f}   TP/FP/FN {top['tp']}/{top['fp']}/{top['fn']}")
    print(f"solution 1 OOF  final 0.8869   Dice 0.7562   F2 0.9383")
    print(f"  {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="One checkpoint, or several to average as an ensemble")
    ap.add_argument("--oof", action="store_true",
                    help="Score all 5 folds, each by the model that held it out")
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs"),
                    help="Directory holding the five fold checkpoints, for --oof")
    ap.add_argument("--pattern", default="alb_unet_f{f}.pt",
                    help="Fold checkpoint filename template, {f} is the fold index")
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

    if args.oof:
        return run_oof(args, device)

    if not args.ckpt:
        sys.exit("give --ckpt, or --oof to score all five folds")

    nets, ckpts = [], []
    for c in args.ckpt:
        n, k = load_net(c, device)
        nets.append(n)
        ckpts.append(k)

    norms = {k["norm"] for k in ckpts}
    assert len(norms) == 1, f"cannot ensemble across normalisations: {norms}"
    norm = norms.pop()
    val_t = build_transforms(norm)[1] if norm else None

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
