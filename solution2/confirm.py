"""Visual confirmation of the pass/fail disagreements between the two solutions.

    python solution2/confirm.py

The claim under test: out-of-fold, solution 2 at threshold 0.02 misses 27
failing specimens where solution 1 at 0.4 misses 37 - so it catches ~10 real
defects the submission model does not. That is a number in a table. This
renders the specimens behind it so it can be checked by eye.

Unlike compare.py, which shows raw argmax, every mask here is produced at the
model's actual operating threshold and passed through the same despeckling the
scored pipeline uses. The severity printed on each tile is evaluation.py's own
compute_max_severity, and the PASS/FAIL verdict is its 25 micron line. What you
see is what was scored.

Out-of-fold throughout: each image is predicted by the one model of each family
that never trained on the micrograph it came from.
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

from data import (VAL_EVERY, VOID_CLASS, index_training, load_image,  # noqa: E402
                  load_mask, to_tensor)
from evaluation import SEVERITY_THRESHOLD, TooManyRegionsError, compute_max_severity  # noqa: E402
from model import build  # noqa: E402
from pipeline import build_transforms  # noqa: E402
from predict import to_mask  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "confirm"
LUT = np.array([[45, 45, 60], [205, 200, 190], [225, 55, 55]], np.uint8)


def load_family(runs, pattern, device):
    """The five fold models of one solution, indexed by the fold they held out."""
    nets = {}
    for f in range(VAL_EVERY):
        path = Path(runs) / pattern.format(f=f)
        if not path.exists():
            sys.exit(f"missing {path}")
        ckpt = torch.load(path, map_location=device)
        net, _ = build(ckpt.get("arch", "unet"), ckpt.get("base", 32),
                       ckpt.get("depth", 4), ckpt.get("chroma", False))
        net = net.to(device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        # Solution 2 normalises in its Compose; solution 1 feeds raw [0,1].
        tf = build_transforms(ckpt["norm"])[1] if "norm" in ckpt else None
        nets[f] = (net, tf)
    return nets


@torch.no_grad()
def void_and_base(entry, img, device):
    """(P(void), matrix/fibre base) for one model on one image."""
    net, tf = entry
    h, w = img.shape[:2]
    if tf is not None:
        x = tf(image=img)["image"][None].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = net(x).float().sigmoid()[0]
    else:
        x = to_tensor(img)[None].to(device)
        ph, pw = (-h) % 16, (-w) % 16
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = net(x).float().softmax(dim=1)[0]
    p = p[:, :h, :w].cpu().numpy()
    return p[VOID_CLASS], (p[1] > p[0]).astype(np.uint8)


def severity_of(mask, um):
    try:
        return compute_max_severity(mask, um)[0]
    except TooManyRegionsError:
        return float("inf")   # judging scores an unscorable prediction as a fail


def tile(mask, label, sev, verdict, w, band=30):
    """One coloured mask with its severity and verdict written above it."""
    canvas = Image.new("RGB", (w, w + band), (255, 255, 255))
    canvas.paste(Image.fromarray(LUT[np.clip(mask, 0, 2)]), (0, band))
    d = ImageDraw.Draw(canvas)
    d.text((2, 2), label[:36], fill=(0, 0, 0))
    # The original image has no verdict; only masks carry a severity.
    if verdict:
        colour = (190, 0, 0) if verdict == "FAIL" else (0, 120, 0)
        text = f"severity {sev:.1f}" if np.isfinite(sev) else "severity over limit"
        d.text((2, 15), f"{text}   {verdict}", fill=colour)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-runs", default=str(REPO / "runs"))
    ap.add_argument("--s2-runs", default=str(REPO / "archive" / "l4-s2" / "solution2" / "runs"))
    ap.add_argument("--s1-threshold", type=float, default=0.4)
    ap.add_argument("--s2-threshold", type=float, default=0.02)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--panels", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("solution 1:", args.s1_runs)
    s1 = load_family(args.s1_runs, "unet_f{f}.pt", device)
    print("solution 2:", args.s2_runs)
    s2 = load_family(args.s2_runs, "alb_unet_f{f}.pt", device)

    out_dir = Path(args.out)
    (out_dir / "s2_catches_s1_misses").mkdir(parents=True, exist_ok=True)
    (out_dir / "s1_catches_s2_misses").mkdir(parents=True, exist_ok=True)

    counts = {"gt_fail": 0, "s1_tp": 0, "s2_tp": 0, "s2_only": 0, "s1_only": 0, "both_miss": 0}
    catches, losses, rows = [], [], []

    for fold in range(VAL_EVERY):
        df = index_training(fold).query("is_val")
        if args.limit:
            df = df.head(args.limit)
        print(f"  fold {fold}: {len(df)} images")

        for _, row in df.iterrows():
            img, um = load_image(row["image"]), row["um_per_pixel"]
            gt = load_mask(row["mask"])
            gt_sev = severity_of(gt, um)
            gt_fail = gt_sev >= SEVERITY_THRESHOLD

            p1, b1 = void_and_base(s1[fold], img, device)
            p2, b2 = void_and_base(s2[fold], img, device)
            m1 = to_mask(p1, b1, args.s1_threshold, args.min_size)
            m2 = to_mask(p2, b2, args.s2_threshold, args.min_size)
            s1_sev, s2_sev = severity_of(m1, um), severity_of(m2, um)
            s1_fail, s2_fail = s1_sev >= SEVERITY_THRESHOLD, s2_sev >= SEVERITY_THRESHOLD

            if not gt_fail:
                continue
            counts["gt_fail"] += 1
            counts["s1_tp"] += s1_fail
            counts["s2_tp"] += s2_fail

            item = (row["stem"], img, gt, gt_sev, m1, s1_sev, m2, s2_sev)
            if s2_fail and not s1_fail:
                counts["s2_only"] += 1
                catches.append(item)
            elif s1_fail and not s2_fail:
                counts["s1_only"] += 1
                losses.append(item)
            elif not s1_fail:
                counts["both_miss"] += 1

            rows.append({"stem": row["stem"], "fold": fold, "um_per_pixel": um,
                         "gt_severity": round(gt_sev, 2),
                         "s1_severity": round(s1_sev, 2), "s1_fail": s1_fail,
                         "s2_severity": round(s2_sev, 2), "s2_fail": s2_fail})

    print(f"\nfailing specimens in ground truth: {counts['gt_fail']}")
    print(f"  solution 1 @ {args.s1_threshold} caught {counts['s1_tp']}, "
          f"missed {counts['gt_fail'] - counts['s1_tp']}")
    print(f"  solution 2 @ {args.s2_threshold} caught {counts['s2_tp']}, "
          f"missed {counts['gt_fail'] - counts['s2_tp']}")
    print(f"\n  solution 2 catches, solution 1 misses : {counts['s2_only']}")
    print(f"  solution 1 catches, solution 2 misses : {counts['s1_only']}")
    print(f"  both miss                             : {counts['both_miss']}")

    with open(out_dir / "disagreements.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Biggest true defects first - a specimen the submission model misses is
    # more damning the more obviously it was failing.
    for bucket, items in [("s2_catches_s1_misses", catches), ("s1_catches_s2_misses", losses)]:
        items.sort(key=lambda t: -t[3])
        for stem, img, gt, gt_sev, m1, s1_sev, m2, s2_sev in items[:args.panels]:
            w = img.shape[1]
            band = 30
            tiles = [tile(np.full(img.shape[:2], -1), "original", np.nan, "", w),
                     tile(gt, "ground truth", gt_sev, "FAIL" if gt_sev >= SEVERITY_THRESHOLD else "pass", w),
                     tile(m1, f"solution 1 @ {args.s1_threshold}", s1_sev,
                          "FAIL" if s1_sev >= SEVERITY_THRESHOLD else "pass", w),
                     tile(m2, f"solution 2 @ {args.s2_threshold}", s2_sev,
                          "FAIL" if s2_sev >= SEVERITY_THRESHOLD else "pass", w)]
            tiles[0].paste(Image.fromarray(img), (0, band))
            gap = 3
            canvas = Image.new("RGB", (4 * w + 3 * gap, w + band), (255, 255, 255))
            for i, t in enumerate(tiles):
                canvas.paste(t, (i * (w + gap), 0))
            canvas.save(out_dir / bucket / f"{gt_sev:08.1f}_{stem}.png")

    print(f"\npanels -> {out_dir}")
    print(f"  s2_catches_s1_misses/  the defects the submission model lets through")
    print(f"  s1_catches_s2_misses/  what the low threshold costs in return")
    print(f"table -> {out_dir / 'disagreements.csv'}")


if __name__ == "__main__":
    main()
