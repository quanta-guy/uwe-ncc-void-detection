"""Final acceptance check: 20 random held-out images, everything measured.

    python solution2/final_check.py
    python solution2/final_check.py --n 20 --seed 0 --pool void

For each sampled image, three things against ground truth:

  severity   evaluation.py's own compute_max_severity on prediction and truth,
             plus the pass/fail call at 25um that judging actually scores
  area       total void area in um2, predicted against true
  regions    region-level correspondence - how many true void regions were
             found, how many predictions have no true void under them

Out-of-fold throughout: each image is predicted by the one fold model that
never trained on its micrograph, so nothing here is a model grading its own
training data.

Sampling defaults to void-containing images (--pool void). All three quantities
are undefined or trivially zero on an image with no defect, and a uniform draw
over the full split would be ~68% empty. --pool all gives the uniform draw and
reports the false-alarm behaviour instead.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import VAL_EVERY, VOID_CLASS, index_training, load_image, load_mask  # noqa: E402
from evaluation import SEVERITY_THRESHOLD  # noqa: E402
from confirm import load_family, severity_of, void_and_base  # noqa: E402
from locate import match, regions as void_regions  # noqa: E402
from predict import to_mask  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "final_check"
LUT = np.array([[45, 45, 60], [205, 200, 190], [225, 55, 55]], np.uint8)


def sample(pool, n, seed):
    """n random held-out images, with the fold that holds each one out."""
    rows = []
    for f in range(VAL_EVERY):
        df = index_training(f).query("is_val")
        for _, r in df.iterrows():
            rows.append((f, r))
    rng = np.random.default_rng(seed)

    if pool == "void":
        keep = [(f, r) for f, r in rows if (load_mask(r["mask"]) == VOID_CLASS).any()]
    else:
        keep = rows
    idx = rng.choice(len(keep), size=min(n, len(keep)), replace=False)
    return [keep[i] for i in sorted(idx)]


def tile(pic, label, lines, w, band, is_image=False):
    canvas = Image.new("RGB", (w, w + band), (255, 255, 255))
    canvas.paste(Image.fromarray((pic if is_image else LUT[np.clip(pic, 0, 2)]).astype(np.uint8)),
                 (0, band))
    d = ImageDraw.Draw(canvas)
    d.text((2, 2), label[:44], fill=(0, 0, 0))
    for i, (text, colour) in enumerate(lines):
        d.text((2, 14 + 12 * i), text[:52], fill=colour)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--pattern", default="unet_f{f}.pt")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", choices=["void", "all"], default="void")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = load_family(args.runs, args.pattern, device)
    picks = sample(args.pool, args.n, args.seed)
    print(f"\n{len(picks)} random held-out images (seed {args.seed}, pool '{args.pool}'), "
          f"threshold {args.threshold}\n")

    out_dir = Path(args.out)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    rows = []

    hdr = (f"{'image':30} {'gt_sev':>8} {'pred_sev':>9} {'verdict':>14} "
           f"{'gt_area':>9} {'pred_area':>10} {'regions':>10} {'dice':>6}")
    print(hdr)
    print("-" * len(hdr))

    for fold, r in picks:
        img, um = load_image(r["image"]), r["um_per_pixel"]
        gt = load_mask(r["mask"])
        p_void, base = void_and_base(nets[fold], img, device)
        pred = to_mask(p_void, base, args.threshold, args.min_size)

        gt_sev, pr_sev = severity_of(gt, um), severity_of(pred, um)
        gt_fail, pr_fail = gt_sev >= SEVERITY_THRESHOLD, pr_sev >= SEVERITY_THRESHOLD

        gt_v, pr_v = gt == VOID_CLASS, pred == VOID_CLASS
        px2 = um * um
        gt_area, pr_area = gt_v.sum() * px2, pr_v.sum() * px2
        inter = (gt_v & pr_v).sum()
        dice = 2 * inter / (gt_v.sum() + pr_v.sum()) if (gt_v.sum() + pr_v.sum()) else 1.0

        pairs, n_gt, n_pred = match(gt, pred, um)
        found = sum(p["detected"] for p in pairs)
        # Predicted regions with no ground-truth void under them.
        matched_pred = {p["pred"] for p in pairs if p["detected"]}
        spurious = n_pred - len(matched_pred)

        verdict = ("both FAIL" if gt_fail and pr_fail else
                   "both pass" if not gt_fail and not pr_fail else
                   "MISS" if gt_fail else "FALSE ALARM")
        print(f"{r['stem'][:30]:30} {gt_sev:8.1f} {pr_sev:9.1f} {verdict:>14} "
              f"{gt_area:9.0f} {pr_area:10.0f} {found:4d}/{n_gt:<2d}+{spurious:<2d} {dice:6.3f}")

        rows.append({"stem": r["stem"], "fold": fold, "um_per_pixel": um,
                     "gt_severity": round(gt_sev, 2), "pred_severity": round(pr_sev, 2),
                     "gt_fail": gt_fail, "pred_fail": pr_fail, "verdict": verdict,
                     "gt_area_um2": round(float(gt_area), 1),
                     "pred_area_um2": round(float(pr_area), 1),
                     "gt_regions": n_gt, "regions_found": found,
                     "pred_regions": n_pred, "spurious_regions": spurious,
                     "dice": round(float(dice), 4)})

        w, band = img.shape[1], 42
        red, green, grey = (190, 0, 0), (0, 120, 0), (70, 70, 70)
        panel = [
            tile(img, r["stem"], [(f"{um:.3f} um/px  fold {fold}", grey)], w, band, True),
            tile(gt, "ground truth",
                 [(f"severity {gt_sev:.1f}  {'FAIL' if gt_fail else 'PASS'}",
                   red if gt_fail else green),
                  (f"{n_gt} regions  {gt_area:.0f} um2", grey)], w, band),
            tile(pred, f"prediction @ {args.threshold}",
                 [(f"severity {pr_sev:.1f}  {'FAIL' if pr_fail else 'PASS'}",
                   red if pr_fail else green),
                  (f"{found}/{n_gt} found, {spurious} spurious  dice {dice:.2f}", grey)],
                 w, band),
        ]
        gap = 3
        canvas = Image.new("RGB", (3 * w + 2 * gap, w + band), (255, 255, 255))
        for i, t in enumerate(panel):
            canvas.paste(t, (i * (w + gap), 0))
        canvas.save(out_dir / "panels" / f"{'MISS_' if verdict == 'MISS' else ''}{r['stem']}.png")

    d = rows
    agree = sum(r["gt_fail"] == r["pred_fail"] for r in d)
    misses = [r for r in d if r["verdict"] == "MISS"]
    alarms = [r for r in d if r["verdict"] == "FALSE ALARM"]
    sev_ratio = np.array([r["pred_severity"] / r["gt_severity"]
                          for r in d if r["gt_severity"] > 0])
    area_ratio = np.array([r["pred_area_um2"] / r["gt_area_um2"]
                           for r in d if r["gt_area_um2"] > 0])
    tot_gt_reg = sum(r["gt_regions"] for r in d)
    tot_found = sum(r["regions_found"] for r in d)
    tot_spur = sum(r["spurious_regions"] for r in d)

    print(f"\n{'=' * 72}")
    print(f"pass/fail agreement   {agree}/{len(d)}  ({agree / len(d):.0%})   "
          f"{len(misses)} missed, {len(alarms)} false alarms")
    print(f"severity  pred/true   median {np.median(sev_ratio):.2f}   "
          f"min {sev_ratio.min():.2f}   max {sev_ratio.max():.2f}")
    print(f"area      pred/true   median {np.median(area_ratio):.2f}   "
          f"min {area_ratio.min():.2f}   max {area_ratio.max():.2f}")
    print(f"regions   {tot_found}/{tot_gt_reg} true regions found "
          f"({tot_found / max(tot_gt_reg, 1):.0%}), {tot_spur} spurious")
    print(f"dice      median {np.median([r['dice'] for r in d]):.3f}")
    if misses:
        print(f"\nmissed failing specimens (panels prefixed MISS_):")
        for r in misses:
            print(f"  {r['stem']}: true {r['gt_severity']:.1f} -> "
                  f"predicted {r['pred_severity']:.1f}, "
                  f"{r['regions_found']}/{r['gt_regions']} regions found")

    with open(out_dir / "final_check.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    print(f"\npanels -> {out_dir / 'panels'}   (original | truth | prediction)")
    print(f"table  -> {out_dir / 'final_check.csv'}")


if __name__ == "__main__":
    main()
