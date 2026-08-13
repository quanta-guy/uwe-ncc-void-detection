"""Write side-by-side overlay panels for visual inspection.

    python src/overlay.py --split test --ckpt runs/unet_e20.pt runs/unet_e12.pt runs/unet.pt
    python src/overlay.py --split val  --ckpt ...        # worst errors first

Test panels are image | prediction, since there is no mask to compare to.
Val panels are image | ground truth | prediction, ordered by how badly the
severity call is wrong - so the first files in the folder are the ones worth
looking at, not a random sample that all look fine.

Void is drawn red, ground-truth void green. Fibre and matrix are left alone;
neither is scored and tinting them only obscures the class that matters.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data import VOID_CLASS, index_test, index_training, load_image, load_mask  # noqa: E402
from evaluation import SEVERITY_THRESHOLD, compute_max_severity, dice_void  # noqa: E402
from predict import load_nets, to_mask, void_prob  # noqa: E402

SCALE = 2
BAR = 22
PRED_RGB = (255, 60, 60)
GT_RGB = (60, 220, 60)


def _tint(img, mask, rgb, alpha=0.45):
    out = img.astype(np.float32).copy()
    sel = mask == VOID_CLASS
    out[sel] = (1 - alpha) * out[sel] + alpha * np.array(rgb, dtype=np.float32)
    return out.astype(np.uint8)


def _panel(tiles, captions, title):
    """Tiles left to right, each captioned, with one title bar on top."""
    h, w = tiles[0].shape[:2]
    sw, sh = w * SCALE, h * SCALE
    canvas = Image.new("RGB", (sw * len(tiles), sh + 2 * BAR), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(235, 235, 235))

    for i, (tile, cap) in enumerate(zip(tiles, captions)):
        canvas.paste(Image.fromarray(tile).resize((sw, sh), Image.NEAREST), (i * sw, BAR))
        draw.text((i * sw + 6, BAR + sh + 4), cap, fill=(200, 200, 200))

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[str(REPO / "runs" / "unet.pt")])
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--fold", type=int, default=0,
                    help="Which fold's held-out images to show; must match the model's")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = load_nets(args.ckpt, device)

    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    # The fold must match the checkpoint, or a fold model is shown images it
    # trained on and the panels prove nothing.
    df = index_test() if args.split == "test" else index_training(args.fold).query("is_val")

    rows = []
    for _, r in df.iterrows():
        img = load_image(r["image"])
        p_void, base = void_prob(nets, img, device)
        pred = to_mask(p_void, base, args.threshold, args.min_size)
        sev, _ = compute_max_severity(pred, r["um_per_pixel"])
        rec = {"stem": r["stem"], "img": img, "pred": pred, "sev": sev,
               "call": "FAIL" if sev >= SEVERITY_THRESHOLD else "pass"}

        if args.split == "val":
            gt = load_mask(r["mask"])
            gt_sev, gt_n = compute_max_severity(gt, r["um_per_pixel"])
            if gt_n == 0 and (pred == VOID_CLASS).sum() == 0:
                continue  # nothing to look at: no void either side
            rec |= {"gt": gt, "gt_sev": gt_sev,
                    "gt_call": "FAIL" if gt_sev >= SEVERITY_THRESHOLD else "pass",
                    "dice": dice_void(gt, pred)}
            # Rank by disagreement first, then by Dice - so false negatives and
            # false positives sort to the top of the folder.
            rec["rank"] = (rec["gt_call"] == rec["call"], rec["dice"])
        else:
            rec["rank"] = ((pred == VOID_CLASS).sum() == 0, r["stem"])
        rows.append(rec)

    rows.sort(key=lambda r: r["rank"])

    for i, r in enumerate(rows[:args.limit], 1):
        if args.split == "val":
            verdict = "OK " if r["gt_call"] == r["call"] else \
                      ("FN " if r["gt_call"] == "FAIL" else "FP ")
            canvas = _panel(
                [r["img"], _tint(r["img"], r["gt"], GT_RGB), _tint(r["img"], r["pred"], PRED_RGB)],
                ["image",
                 f"truth  sev {r['gt_sev']:.1f}  {r['gt_call']}",
                 f"pred   sev {r['sev']:.1f}  {r['call']}"],
                f"{verdict} {r['stem']}   Dice {r['dice']:.3f}")
            name = f"{i:02d}_{verdict.strip()}_{r['stem']}.png"
        else:
            canvas = _panel(
                [r["img"], _tint(r["img"], r["pred"], PRED_RGB)],
                ["image", f"pred   sev {r['sev']:.1f}  {r['call']}"],
                f"{r['stem']}   {(r['pred'] == VOID_CLASS).sum()} void px")
            name = f"{i:02d}_{r['call']}_{r['stem']}.png"
        canvas.save(out_dir / name)

    print(f"wrote {min(len(rows), args.limit)} panels to {out_dir}")
    if args.split == "val":
        bad = sum(1 for r in rows if r["gt_call"] != r["call"])
        print(f"({bad} disagreements in {len(rows)} void-relevant val images, sorted first)")


if __name__ == "__main__":
    main()
