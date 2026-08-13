"""Are the predicted voids in the RIGHT PLACE, or merely the right amount?

    python solution2/locate.py --ckpt runs/unet_f0.pt --threshold 0.4

Dice and severity can both look healthy while the mask is subtly misplaced.
Severity is computed from geometry alone - lengths and areas of whatever
regions exist - so a mask shifted a few pixels, or one that finds the right
NUMBER of voids in the wrong places, scores about the same as a correct one.
Nothing in evaluation.py would notice, because judging never asks where.

Three questions this answers that pixel Dice does not:

1. **Region-level detection.** Of the ground truth's separate void regions, how
   many have any predicted void on top of them? A model can hit 0.75 Dice by
   segmenting big voids beautifully and missing every small one, and per-pixel
   Dice hides that behind the area weighting.

2. **Systematic offset.** Averaged over every matched pair, is the predicted
   centroid displaced in a consistent direction? Random error averages to zero;
   a real offset does not. A non-zero mean displacement is a padding,
   resize or transform bug, and it is exactly the kind that produces plausible
   masks and quietly costs score.

3. **Centroid displacement.** For regions that are found, how far off is the
   centre, in microns rather than pixels, so it can be read against the 40um
   clustering distance that decides how voids group into severity.

Scoring is untouched: this imports evaluation.py's own constants and never
recomputes anything it already defines.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import center_of_mass, label

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import VOID_CLASS, index_training, load_image, load_mask  # noqa: E402
from evaluation import MERGE_DISTANCE_UM  # noqa: E402  - the judge's own clustering distance
from evaluate import class_prob, load_net  # noqa: E402
from pipeline import build_transforms  # noqa: E402
from predict import to_mask  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "locate"


def regions(mask):
    """Connected void components as (labels, count). 8-connectivity, matching
    the way evaluation.py groups pixels before measuring them."""
    return label(mask == VOID_CLASS, structure=np.ones((3, 3), int))


def match(gt_mask, pred_mask, um):
    """Pair up ground-truth and predicted void regions by overlap.

    Returns (per-pair records, n_gt, n_pred). A ground-truth region counts as
    detected if ANY predicted void pixel lands on it - a deliberately generous
    test, because the question here is localisation, not boundary quality.
    """
    gt_lab, n_gt = regions(gt_mask)
    pr_lab, n_pred = regions(pred_mask)
    pairs = []

    for g in range(1, n_gt + 1):
        gsel = gt_lab == g
        overlapping = np.unique(pr_lab[gsel])
        overlapping = overlapping[overlapping > 0]
        if len(overlapping) == 0:
            pairs.append({"gt": g, "pred": 0, "detected": False, "gt_px": int(gsel.sum()),
                          "pred_px": 0, "dy_um": np.nan, "dx_um": np.nan, "dist_um": np.nan})
            continue
        # Several predictions can land on one truth region when a void is
        # fragmented; take the largest as the match.
        best = max(overlapping, key=lambda p: int(((pr_lab == p) & gsel).sum()))
        psel = pr_lab == best
        gy, gx = center_of_mass(gsel)
        py, px = center_of_mass(psel)
        dy, dx = (py - gy) * um, (px - gx) * um
        pairs.append({"gt": g, "pred": int(best), "detected": True,
                      "gt_px": int(gsel.sum()), "pred_px": int(psel.sum()),
                      "dy_um": dy, "dx_um": dx, "dist_um": float(np.hypot(dy, dx))})

    return pairs, n_gt, n_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets, ckpts = zip(*[load_net(c, device) for c in args.ckpt])
    norm = ckpts[0]["norm"]
    val_t = build_transforms(norm)[1] if norm else None

    fold = args.fold if args.fold is not None else ckpts[0]["fold"]
    df = index_training(fold).query("is_val")
    if args.limit:
        df = df.head(args.limit)
    print(f"\nfold {fold}: {len(df)} held-out images, threshold {args.threshold}\n")

    rows, all_pairs = [], []
    tot_gt = tot_pred = tot_detected = 0

    for _, row in df.iterrows():
        img = load_image(row["image"])
        gt = load_mask(row["mask"])
        if not (gt == VOID_CLASS).any():
            # Still count predicted regions here: every one is a false region.
            p = class_prob(list(nets), val_t, img, device)
            pred = to_mask(p[VOID_CLASS], (p[1] > p[0]).astype(np.uint8),
                           args.threshold, args.min_size)
            tot_pred += regions(pred)[1]
            continue

        p = class_prob(list(nets), val_t, img, device)
        pred = to_mask(p[VOID_CLASS], (p[1] > p[0]).astype(np.uint8),
                       args.threshold, args.min_size)
        pairs, n_gt, n_pred = match(gt, pred, row["um_per_pixel"])

        tot_gt += n_gt
        tot_pred += n_pred
        tot_detected += sum(p_["detected"] for p_ in pairs)
        all_pairs += pairs
        for p_ in pairs:
            rows.append({"stem": row["stem"], "um_per_pixel": row["um_per_pixel"], **p_})

    found = [p for p in all_pairs if p["detected"]]
    dists = np.array([p["dist_um"] for p in found])
    dys = np.array([p["dy_um"] for p in found])
    dxs = np.array([p["dx_um"] for p in found])
    gt_px = np.array([p["gt_px"] for p in found], float)
    pred_px = np.array([p["pred_px"] for p in found], float)

    print(f"{'ground-truth void regions':38} {tot_gt}")
    print(f"{'  detected (any overlap)':38} {tot_detected}  "
          f"({tot_detected / max(tot_gt, 1):.1%} region-level recall)")
    print(f"{'predicted void regions':38} {tot_pred}")

    print(f"\ncentroid displacement of detected regions, microns")
    print(f"{'  median':38} {np.median(dists):.2f}")
    print(f"{'  p90':38} {np.percentile(dists, 90):.2f}")
    print(f"{'  max':38} {dists.max():.2f}")
    print(f"  for scale, judging merges voids closer than {MERGE_DISTANCE_UM}um")

    # The decisive integrity test. Random localisation error cancels; a
    # systematic transform bug does not. Compare the mean against the standard
    # error of the mean - anything past a few sigma is a real shift.
    n = len(found)
    sem_y, sem_x = dys.std() / np.sqrt(n), dxs.std() / np.sqrt(n)
    print(f"\nsystematic offset (mean signed displacement over {n} regions)")
    print(f"{'  dy':38} {dys.mean():+.3f} um   (sem {sem_y:.3f}, "
          f"{abs(dys.mean()) / max(sem_y, 1e-9):.1f} sigma)")
    print(f"{'  dx':38} {dxs.mean():+.3f} um   (sem {sem_x:.3f}, "
          f"{abs(dxs.mean()) / max(sem_x, 1e-9):.1f} sigma)")
    shifted = max(abs(dys.mean()) / max(sem_y, 1e-9), abs(dxs.mean()) / max(sem_x, 1e-9)) > 5
    print(f"  -> {'SYSTEMATIC SHIFT - suspect a padding/resize/transform bug' if shifted else 'no systematic shift; error is random, as it should be'}")

    print(f"\narea of matched predictions vs truth")
    ratio = pred_px / gt_px.clip(min=1)
    print(f"{'  median pred/gt area':38} {np.median(ratio):.3f}")
    print(f"{'  p10 / p90':38} {np.percentile(ratio, 10):.3f} / {np.percentile(ratio, 90):.3f}")

    # Detection against region size is the thing per-pixel Dice hides most.
    print(f"\nregion-level recall by true region size (pixels)")
    sizes = np.array([p["gt_px"] for p in all_pairs])
    det = np.array([p["detected"] for p in all_pairs])
    for lo, hi in [(0, 10), (10, 25), (25, 100), (100, 400), (400, 10**9)]:
        sel = (sizes >= lo) & (sizes < hi)
        if sel.any():
            tag = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            print(f"  {tag:>10} px  {sel.sum():5d} regions  {det[sel].mean():6.1%} found")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.ckpt[0]).stem
    with open(out_dir / f"{name}_regions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-region table -> {out_dir / f'{name}_regions.csv'}")


if __name__ == "__main__":
    main()
