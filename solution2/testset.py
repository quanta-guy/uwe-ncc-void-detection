"""Run the best models on the Test set and show what they predict.

    python solution2/testset.py --ckpt runs/unet_f0.pt:0.4 \
                                       solution2/runs/micronet_e45_f0.pt:0.1

Each --ckpt is path:threshold, because the operating point is fitted per model
and comparing models at a shared threshold would compare fitted knobs instead.

**The Test set has no ground-truth masks.** index_test() returns mask=None -
predicting them is the task. So nothing here can be checked against truth, and
no accuracy claim is possible from this script. What it does give is what a
reviewer actually needs before submitting:

  - the severity evaluation.py would compute for each prediction, and the
    pass/fail call at 25um
  - the anatomy behind that number: region count, cluster count, the largest
    single diameter, and total void area
  - whether the models agree with each other, which is the only cross-check
    available without labels

Area-against-truth lives in locate.py, which runs on held-out training folds
where masks exist. Median predicted region area there is 1.07x truth.

All severity numbers come from evaluation.py itself, unchanged.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import VOID_CLASS, index_test, load_image  # noqa: E402
from evaluation import (SEVERITY_THRESHOLD, TooManyRegionsError,  # noqa: E402
                        _VoidRegions, compute_max_severity, merge_regions)
from evaluate import class_prob, load_net  # noqa: E402
from pipeline import build_transforms  # noqa: E402
from predict import to_mask  # noqa: E402
sys.path.insert(0, str(REPO / 'solution3'))
from data3 import resample  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "testset"
LUT = np.array([[45, 45, 60], [205, 200, 190], [225, 55, 55]], np.uint8)


def anatomy(mask, um):
    """Severity plus the quantities that produce it, per evaluation.py."""
    try:
        sev, n_clusters = compute_max_severity(mask, um)
    except TooManyRegionsError:
        return {"severity": float("inf"), "clusters": -1, "regions": -1,
                "max_diameter_um": np.nan, "void_area_um2": np.nan,
                "void_px": int((mask == VOID_CLASS).sum())}

    regions = _VoidRegions(mask)
    if regions.n == 0:
        return {"severity": 0.0, "clusters": 0, "regions": 0,
                "max_diameter_um": 0.0, "void_area_um2": 0.0, "void_px": 0}

    # Report the anatomy of the worst cluster - the one that sets the score.
    groups = merge_regions(regions, um)
    worst = max(groups, key=lambda g: sum(regions.diameter(i) * um for i in g)
                + 0.5 * np.sqrt(sum(regions.areas[i] * um * um for i in g)))
    return {
        "severity": float(sev), "clusters": int(n_clusters), "regions": int(regions.n),
        "max_diameter_um": float(max(regions.diameter(i) * um for i in worst)),
        "void_area_um2": float(sum(regions.areas[i] * um * um for i in range(regions.n))),
        "void_px": int((mask == VOID_CLASS).sum()),
    }


def tile(img_or_mask, label, info, w, band=42, is_image=False):
    canvas = Image.new("RGB", (w, w + band), (255, 255, 255))
    pic = img_or_mask if is_image else LUT[np.clip(img_or_mask, 0, 2)]
    canvas.paste(Image.fromarray(pic.astype(np.uint8)), (0, band))
    d = ImageDraw.Draw(canvas)
    d.text((2, 2), label[:40], fill=(0, 0, 0))
    if info:
        sev = info["severity"]
        fail = sev >= SEVERITY_THRESHOLD
        colour = (190, 0, 0) if fail else (0, 120, 0)
        txt = "severity over limit" if not np.isfinite(sev) else f"severity {sev:.1f}"
        d.text((2, 14), f"{txt}  {'FAIL' if fail else 'PASS'}", fill=colour)
        d.text((2, 27), f"{info['regions']}r/{info['clusters']}c  "
                        f"dia {info['max_diameter_um']:.0f}um  "
                        f"area {info['void_area_um2']:.0f}um2", fill=(70, 70, 70))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="path[+path...][:threshold]. '+' averages checkpoints as one "
                         "ensemble, which is what actually ships - on the Test set every "
                         "fold model is usable because none of these micrographs were "
                         "in training at all.")
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = []
    for c in args.ckpt:
        paths, _, thr = c.rpartition(":")
        if not paths:              # no colon given
            paths, thr = c, "0.4"
        members = paths.split("+")
        nets, ckpts = zip(*[load_net(p, device) for p in members])
        norms = {k["norm"] for k in ckpts}
        assert len(norms) == 1, f"cannot ensemble across normalisations: {norms}"
        norm = norms.pop()
        # Solution 3 trained on images resampled to a canonical um/pixel, so it
        # must be fed the same way. Feeding it native resolution would present
        # fibres at 2-36px when it only ever saw 6-7px.
        targets = {k.get("target_um") for k in ckpts}
        assert len(targets) == 1, f"cannot ensemble across spacings: {targets}"
        name = Path(members[0]).stem
        if len(members) > 1:
            name = f"{name.rstrip('01234fs_')}_ens{len(members)}"
        specs.append({"name": name, "nets": list(nets), "thr": float(thr),
                      "target_um": targets.pop(),
                      "tf": build_transforms(norm)[1] if norm else None})
    print()

    df = index_test()
    if args.limit:
        df = df.head(args.limit)
    print(f"Test set: {len(df)} images, no ground-truth masks\n")

    out_dir = Path(args.out)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    rows, verdicts = [], {s["name"]: [] for s in specs}

    for _, row in df.iterrows():
        img, um = load_image(row["image"]), row["um_per_pixel"]
        tiles = [tile(img, f"{row['stem']}", None, img.shape[1], is_image=True)]

        for s in specs:
            if s["target_um"]:
                # Predict at the canonical spacing, then bring the probability
                # map back to the native grid - severity is in microns, so a
                # mask on the wrong grid mis-measures a void it located.
                r_img, _, _ = resample(img, None, um, s["target_um"])
                p = class_prob(s["nets"], s["tf"], r_img, device)
                if p.shape[1:] != img.shape[:2]:
                    p = np.stack([cv2.resize(c, (img.shape[1], img.shape[0]),
                                             interpolation=cv2.INTER_LINEAR) for c in p])
            else:
                p = class_prob(s["nets"], s["tf"], img, device)
            mask = to_mask(p[VOID_CLASS], (p[1] > p[0]).astype(np.uint8),
                           s["thr"], args.min_size)
            info = anatomy(mask, um)
            verdicts[s["name"]].append(info["severity"] >= SEVERITY_THRESHOLD)
            rows.append({"stem": row["stem"], "model": s["name"],
                         "threshold": s["thr"], "um_per_pixel": um, **info,
                         "verdict": "FAIL" if info["severity"] >= SEVERITY_THRESHOLD else "PASS"})
            tiles.append(tile(mask, f"{s['name'][:28]} @{s['thr']}", info, img.shape[1]))

        gap = 3
        w, h = tiles[0].size
        canvas = Image.new("RGB", (len(tiles) * w + (len(tiles) - 1) * gap, h), (255, 255, 255))
        for i, t in enumerate(tiles):
            canvas.paste(t, (i * (w + gap), 0))
        canvas.save(out_dir / "panels" / f"{row['stem']}.png")

    names = [s["name"] for s in specs]
    print(f"{'model':34} {'threshold':>9} {'FAIL':>6} {'PASS':>6}")
    for n in names:
        v = np.array(verdicts[n])
        thr = next(s["thr"] for s in specs if s["name"] == n)
        print(f"{n[:34]:34} {thr:9.2f} {v.sum():6d} {(~v).sum():6d}")

    if len(names) > 1:
        a, b = np.array(verdicts[names[0]]), np.array(verdicts[names[1]])
        agree = (a == b).mean()
        print(f"\nthe two models agree on {agree:.1%} of Test images "
              f"({int((a != b).sum())} disagreements)")
        print("  with no labels here, disagreement is the only signal available -")
        print("  those images are the ones worth a human look before submitting")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"    {df.iloc[i]['stem']}: {names[0]}="
                      f"{'FAIL' if x else 'PASS'}  {names[1]}={'FAIL' if y else 'PASS'}")

    with open(out_dir / "testset.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    print(f"\npanels -> {out_dir / 'panels'}")
    print(f"table  -> {out_dir / 'testset.csv'}")


if __name__ == "__main__":
    main()
