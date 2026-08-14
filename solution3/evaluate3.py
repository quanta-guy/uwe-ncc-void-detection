"""Solution 3 scoring: nested threshold selection, no optimism.

    python solution3/evaluate3.py --oof          # all 5 folds, nested tuning
    python solution3/evaluate3.py --ckpt solution3/runs/s3_unet_f0_s0.pt

Two problems with how solution 1 reported 0.8869, both fixed here.

**Non-nested tuning.** That sweep chose threshold and min_size on the same
out-of-fold predictions it then reported. Selecting a hyperparameter on the
data you report is optimistic by construction. Here the knobs are chosen on an
inner split of the training micrographs and applied unchanged to the untouched
outer fold, so the reported number involves no selection on the data it
describes.

**Ensemble scored on data it trained on.** bench.py graded the five-fold
ensemble on fold 0, where four of its five members had trained. That number was
invalid. This never ensembles across folds for scoring - each outer fold is
scored only by models that never saw it.

Inference runs at the canonical spacing the model was trained on, then the
probability map is resized back to NATIVE resolution before thresholding.
Severity is measured in microns through um_per_pixel, so a mask on the wrong
grid reports the wrong severity for a void it located perfectly.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
from data import VOID_CLASS, load_image, load_mask, to_tensor  # noqa: E402
from data3 import CANONICAL_UM_PER_PX, N_FOLDS, balanced_folds, index3, resample  # noqa: E402
from model import build  # noqa: E402
from predict import _sweep, to_mask  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MIN_SIZES = [2, 4, 10]


def load_ckpt(path, device):
    """Rebuild a checkpoint. `target_um` comes from the checkpoint, never a flag.

    A model trained at one spacing and scored at another still runs and still
    produces plausible maps - it just sees fibres at the wrong size and scores
    worse for no visible reason. That is the same failure as evaluating under
    the wrong normalisation, and the only safe place for the answer is the
    checkpoint itself.
    """
    ckpt = torch.load(path, map_location=device)
    net, _ = build(ckpt.get("arch", "unet"), ckpt.get("base", 32),
                   ckpt.get("depth", 4), ckpt.get("chroma", False))
    net = net.to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    if "target_um" not in ckpt:
        sys.exit(f"{Path(path).name} has no target_um - retrain, or it cannot be "
                 f"scored at a known spacing")
    return net, ckpt


def resolve_target(ckpts, override=None):
    """The spacing these checkpoints were trained at, with an explicit override.

    Ensembling across spacings is refused rather than silently averaged: the
    members would be looking at different physical scales.
    """
    spacings = {round(float(k["target_um"]), 6) for k in ckpts}
    if len(spacings) != 1:
        sys.exit(f"checkpoints span different training spacings: {sorted(spacings)}")
    trained = spacings.pop()
    if override is not None and abs(override - trained) > 1e-6:
        print(f"  WARNING: --target-um {override} overrides the {trained} these models "
              f"were trained at. Deliberate re-scaling only; otherwise the score is wrong.")
        return override
    return trained


@torch.no_grad()
def predict_native(net, img, um_per_px, device, target=CANONICAL_UM_PER_PX):
    """(P(void), matrix/fibre base) on the image's NATIVE pixel grid.

    Resample up to the canonical spacing the model was trained at, predict,
    then resize the probability back down. Thresholding happens at native
    resolution so severity, which is computed in microns, stays correct.
    """
    h, w = img.shape[:2]
    r_img, _, _ = resample(img, None, um_per_px, target)
    x = to_tensor(r_img)[None].to(device)

    # The U-Net halves resolution four times, so pad up to a multiple of 16.
    ph, pw = (-x.shape[-2]) % 16, (-x.shape[-1]) % 16
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")

    with torch.autocast(device.type, enabled=device.type == "cuda"):
        p = net(x).float().softmax(dim=1)[0, :, :r_img.shape[0], :r_img.shape[1]]
    p = p.cpu().numpy()

    if p.shape[1:] != (h, w):
        p = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in p])
    return p[VOID_CLASS], (p[1] > p[0]).astype(np.uint8)


def collect(net, df, device, target):
    probs, bases, gts, ums, greys = [], [], [], [], []
    for _, row in df.iterrows():
        img = load_image(row["image"])
        p_void, base = predict_native(net, img, row["um_per_pixel"], device, target)
        probs.append(p_void.astype(np.float16))
        bases.append(base)
        gts.append(load_mask(row["mask"]))
        ums.append(row["um_per_pixel"])
        greys.append(img.mean(axis=2).astype(np.uint8))
    return probs, bases, gts, ums, greys


def score_at(cached, threshold, min_size):
    """Score one fixed setting - no search, so no selection on this data."""
    _, rows = _sweep(*cached, [threshold], [min_size], dilations=[0])
    return rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--pattern", default="s3_unet_f{f}_s0.pt")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--target-um", type=float, default=None,
                    help="Override the spacing the checkpoints were trained at. "
                         "Leave unset - the default is read from each checkpoint.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assign, _ = balanced_folds(index3(0))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.oof:
        net, ckpt = load_ckpt(args.ckpt, device)
        target = resolve_target([ckpt], args.target_um)
        df = index3(ckpt["fold"], assignment=assign).query("is_val")
        if args.limit:
            df = df.head(args.limit)
        print(f"  scoring at {target} um/px (trained at {ckpt['target_um']})")
        cached = collect(net, df, device, target)
        best, rows = _sweep(*cached, THRESHOLDS, MIN_SIZES, dilations=[0])
        print(f"\nfold {ckpt['fold']}: {len(df)} held-out originals, "
              f"tuned-on-this-data score {best[0]:.4f} (optimistic - use --oof)")
        return

    # Nested: the knobs come from folds that are not the one being scored, so
    # nothing is selected on the data being reported.
    print("collecting predictions per fold (each by the model that held it out)\n")
    loaded = []
    for f in range(N_FOLDS):
        path = Path(args.runs) / args.pattern.format(f=f)
        if not path.exists():
            sys.exit(f"missing {path}")
        net, ckpt = load_ckpt(path, device)
        assert ckpt["fold"] == f, f"{path.name} says fold {ckpt['fold']}"
        loaded.append((net, ckpt))

    # One spacing for the whole sweep, taken from the checkpoints. Resolving it
    # before any prediction means a mixed-spacing run stops here rather than
    # producing a number that looks fine.
    target = resolve_target([c for _, c in loaded], args.target_um)
    print(f"  scoring at {target} um/px, read from the checkpoints\n")

    per_fold = {}
    for f, (net, ckpt) in enumerate(loaded):
        df = index3(f, assignment=assign).query("is_val")
        if args.limit:
            df = df.head(args.limit)
        print(f"  fold {f}: {len(df)} held-out originals, {df.group.nunique()} micrographs")
        per_fold[f] = collect(net, df, device, target)

    rows_out, finals = [], []
    for f in range(N_FOLDS):
        # Inner selection on the OTHER folds only.
        inner = tuple(sum((list(per_fold[g][k]) for g in range(N_FOLDS) if g != f), [])
                      for k in range(5))
        best_inner, _ = _sweep(*inner, THRESHOLDS, MIN_SIZES, dilations=[0])
        t, m = best_inner[1], best_inner[2]

        outer = score_at(per_fold[f], t, m)
        finals.append(outer["final"])
        rows_out.append({"fold": f, "chosen_threshold": t, "chosen_min_size": m, **outer})
        print(f"\nfold {f}: inner-selected --threshold {t} --min-size {m} "
              f"-> outer final {outer['final']:.4f} "
              f"(Dice {outer['dice']:.4f}, F2 {outer['f2']:.4f}, "
              f"TP/FP/FN {outer['tp']}/{outer['fp']}/{outer['fn']})")

    with open(out_dir / "nested_oof.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    print(f"\n{'=' * 66}")
    print(f"nested out-of-fold final: mean {np.mean(finals):.4f}  "
          f"sd {np.std(finals):.4f}  range {min(finals):.4f}-{max(finals):.4f}")
    print("solution 1 reported 0.8869, tuned on the same OOF data it reported -")
    print("that number is optimistic; this one is not, so they are not directly")
    print("comparable. Compare it against a non-nested sweep of these same models.")
    print(f"  {out_dir / 'nested_oof.csv'}")


if __name__ == "__main__":
    main()
