"""Inspection report: the product view, not the leaderboard view.

    python src/report.py --ckpt runs/unet_f0.pt runs/unet_f1.pt ... --split test

A submission is a folder of masks. An inspector needs three things a mask
does not carry:

  why      which voids drove the severity, and how the number was built, so an
           engineer can argue with it rather than take it on faith
  how sure the five fold-models are independent opinions; where they disagree
           the call is soft, and saying so is the difference between a tool
           that gets trusted and one that gets overridden
  what now a queue ordered by what deserves human eyes, not 32 files in
           alphabetical order

The triage rule is deliberately not the scoring rule. Judging wants a binary
call. An inspector wants to know which binaries are shaky - a specimen at
severity 25.7 against a limit of 25 is not the same evidence as one at 66,
even though both score identically.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data import VOID_CLASS, index_test, index_training, load_image  # noqa: E402
from evaluation import (  # noqa: E402  - reusing judging's own geometry, not reimplementing it
    K_AREA,
    MERGE_DISTANCE_UM,
    SEVERITY_THRESHOLD,
    _VoidRegions,
    merge_regions,
)
from predict import load_nets, to_mask, void_prob  # noqa: E402

# A call within this fraction of the limit is too close to assert on its own.
MARGIN = 0.20
# Mean disagreement between fold models, above which the pixels are contested.
SPREAD_LIMIT = 0.15


def analyse(mask, um_per_px):
    """Severity, and the void group that produced it, itemised."""
    regions = _VoidRegions(mask)
    if regions.n == 0:
        return 0.0, []

    groups = merge_regions(regions, um_per_px)
    scored = []
    for g in groups:
        lengths = [regions.diameter(i) * um_per_px for i in g]
        areas = [regions.areas[i] * um_per_px**2 for i in g]
        scored.append((sum(lengths) + K_AREA * np.sqrt(sum(areas)), lengths, areas))

    sev, lengths, areas = max(scored, key=lambda s: s[0])
    items = sorted(zip(lengths, areas), reverse=True)
    return sev, items


def confidence(spread_map, void_mask):
    """Mean disagreement between models over the pixels they called void."""
    if not void_mask.any():
        return 0.0
    return float(spread_map[void_mask].mean())


def triage(sev, spread, has_void):
    """What an inspector should do next, which is not the same as pass/fail."""
    if spread > SPREAD_LIMIT:
        return "REVIEW", "models disagree on the void extent"
    if abs(sev - SEVERITY_THRESHOLD) <= MARGIN * SEVERITY_THRESHOLD:
        return "REVIEW", f"severity {sev:.1f} within {MARGIN:.0%} of the {SEVERITY_THRESHOLD} limit"
    if sev >= SEVERITY_THRESHOLD:
        return "REJECT", f"severity {sev:.1f} exceeds {SEVERITY_THRESHOLD}"
    if not has_void:
        return "ACCEPT", "no void detected"
    return "ACCEPT", f"severity {sev:.1f} below {SEVERITY_THRESHOLD}"


def _heat(spread):
    """Disagreement as a red heat map, so contested pixels are visible."""
    v = np.clip(spread / 0.35, 0, 1)[..., None]
    return (v * np.array([255, 40, 40]) + (1 - v) * np.array([28, 28, 34])).astype(np.uint8)


def _tint(img, mask, rgb=(255, 60, 60), alpha=0.45):
    out = img.astype(np.float32).copy()
    sel = mask == VOID_CLASS
    out[sel] = (1 - alpha) * out[sel] + alpha * np.array(rgb, dtype=np.float32)
    return out.astype(np.uint8)


def _panel(tiles, captions, title, subtitle):
    h, w = tiles[0].shape[:2]
    sw, sh = w * 2, h * 2
    canvas = Image.new("RGB", (sw * len(tiles), sh + 62), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(240, 240, 240))
    draw.text((8, 22), subtitle, fill=(170, 170, 180))
    for i, (tile, cap) in enumerate(zip(tiles, captions)):
        canvas.paste(Image.fromarray(tile).resize((sw, sh), Image.NEAREST), (i * sw, 40))
        draw.text((i * sw + 8, sh + 44), cap, fill=(200, 200, 200))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--out", default=str(REPO / "results" / "report"))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--min-size", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = load_nets(args.ckpt, device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = index_test() if args.split == "test" else index_training().query("is_val")

    rows = []
    for _, r in df.iterrows():
        img = load_image(r["image"])

        # Each fold model votes separately: the mean is the prediction, the
        # spread is the uncertainty. Running them apart rather than batched
        # costs nothing at this scale and keeps the per-model opinion.
        per_model = [void_prob([n], img, device)[0] for n in nets]
        p_mean = np.mean(per_model, axis=0)
        spread = np.std(per_model, axis=0) if len(nets) > 1 else np.zeros_like(p_mean)

        _, base = void_prob(nets, img, device)
        mask = to_mask(p_mean, base, args.threshold, args.min_size)
        void = mask == VOID_CLASS

        sev, items = analyse(mask, r["um_per_pixel"])
        conf = confidence(spread, void)
        action, reason = triage(sev, conf, void.any())

        rows.append({
            "image_id": r["stem"], "action": action, "reason": reason,
            "severity_um": round(sev, 1), "limit": SEVERITY_THRESHOLD,
            "n_voids_in_defect": len(items), "void_px": int(void.sum()),
            "model_disagreement": round(conf, 3),
        })

        if action != "ACCEPT" or void.any():
            drive = "  ".join(f"L{L:.0f}/A{A:.0f}" for L, A in items[:4])
            canvas = _panel(
                [img, _tint(img, mask), _heat(spread)],
                [f"micrograph  {r['um_per_pixel']} um/px",
                 f"detected void  {int(void.sum())} px",
                 f"model disagreement  {conf:.3f}"],
                f"{r['stem']}    {action}  -  {reason}",
                f"defect: {len(items)} void(s) merged within {MERGE_DISTANCE_UM} um   {drive}")
            canvas.save(out_dir / f"{action}_{r['stem']}.png")

    order = {"REJECT": 0, "REVIEW": 1, "ACCEPT": 2}
    rows.sort(key=lambda r: (order[r["action"]], -r["severity_um"]))

    with open(out_dir / "inspection_queue.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'image_id':34} {'action':7} {'sev':>6} {'disagree':>9}  reason")
    for r in rows:
        if r["action"] == "ACCEPT" and r["void_px"] == 0:
            continue
        print(f"{r['image_id'][:34]:34} {r['action']:7} {r['severity_um']:6.1f} "
              f"{r['model_disagreement']:9.3f}  {r['reason']}")

    counts = {a: sum(1 for r in rows if r["action"] == a) for a in order}
    print(f"\n{len(rows)} specimens -> {counts['REJECT']} reject, "
          f"{counts['REVIEW']} need review, {counts['ACCEPT']} accept")
    print(f"queue + panels: {out_dir}")


if __name__ == "__main__":
    main()
