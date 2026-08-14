"""Render what the submitted model actually predicted, one panel per Test image.

    python tools/test_panels.py                    # -> test_panels/

Three views side by side - original, 3-class mask, void overlay - with the severity
that decides the specimen printed underneath.

Runs the 5-fold ensemble at the submitted operating point (threshold 0.4, min-size 4)
and then **asserts each mask is byte-identical to the corresponding file in
predicted_masks/**. That check is the point: without it these would be pretty pictures
of some nearby model, and a reviewer could not tell. With it, every panel is provably
the submission.

Severity comes from evaluation.py - NCC's own scorer - never recomputed here.

The output folder embeds NCC's micrographs, so it is gitignored. Share it through the
team Drive or the deck, not the public repo.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "frontend" / "tools"))

import torch  # noqa: E402
from data import VOID_CLASS, index_test, load_image  # noqa: E402
from evaluation import SEVERITY_THRESHOLD, TooManyRegionsError, compute_max_severity  # noqa: E402
from predict import load_nets, to_mask, void_prob  # noqa: E402
from severity_geometry import controlling_geometry  # noqa: E402

THRESHOLD, MIN_SIZE = 0.4, 4          # the submitted operating point, see SUBMISSION.md
SCALE = 2                             # 256px micrographs are unreadable on a slide
GAP, PAD = 10, 14
CAPTION_H = 96

#: Matches the frontend: matrix neutral, fibre teal, void red.
LUT = np.array([[38, 42, 48], [0, 133, 139], [213, 31, 38]], np.uint8)
YELLOW = (255, 199, 0)


def font(size, bold=False):
    for name in (("arialbd.ttf", "seguisb.ttf") if bold else ("arial.ttf", "segoeui.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(img, mask, evidence):
    """Original with voids tinted and the controlling cluster ringed.

    The ring is what makes the number auditable - it marks the one cluster whose
    severity became the specimen's score, rather than leaving a reviewer to guess
    which of several voids drove it.
    """
    from scipy.ndimage import binary_dilation

    out = img.astype(np.float32).copy()
    sel = mask == VOID_CLASS
    out[sel] = 0.35 * out[sel] + 0.65 * np.array(LUT[VOID_CLASS], np.float32)

    idx = [v["index"] for v in evidence["voids"]]
    if idx:
        from evaluation import _VoidRegions
        regions = _VoidRegions(mask)
        ctrl = np.zeros(mask.shape, bool)
        for i in idx:
            ctrl[tuple(regions.coords[i].T)] = True
        ring = binary_dilation(ctrl, np.ones((5, 5), bool)) & ~ctrl
        out[ring] = YELLOW
    return out.astype(np.uint8)


def up(arr):
    im = Image.fromarray(arr)
    return im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)


def panel(img, mask, evidence, stem, um, severity):
    tiles = [up(img), up(LUT[np.clip(mask, 0, 2)]), up(overlay(img, mask, evidence))]
    w, h = tiles[0].size
    total_w = 3 * w + 2 * GAP + 2 * PAD
    canvas = Image.new("RGB", (total_w, h + CAPTION_H + 2 * PAD + 22), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    d.text((PAD, PAD - 2), stem, fill=(16, 24, 32), font=font(15, bold=True))
    for n, (t, label) in enumerate(zip(tiles, ("Original", "Segmentation", "Void overlay"))):
        x = PAD + n * (w + GAP)
        canvas.paste(t, (x, PAD + 20))
        d.text((x + 2, PAD + 22), f" {label} ", fill=(255, 255, 255),
               font=font(13, bold=True), stroke_width=3, stroke_fill=(0, 0, 0))

    y = PAD + 20 + h + 12
    fail = severity >= SEVERITY_THRESHOLD
    colour = (190, 20, 26) if fail else (30, 122, 56)
    verdict = "FAIL" if fail else "PASS"
    sev_txt = "over region limit" if not np.isfinite(severity) else f"{severity:.2f} µm"

    d.text((PAD, y), f"Severity {sev_txt}   {verdict}", fill=colour, font=font(21, bold=True))
    d.text((PAD, y + 28),
           f"limit {SEVERITY_THRESHOLD} µm   ·   "
           f"ΣL {evidence['length_term_um']:.2f} µm + 0.5·√(ΣA) {evidence['area_term_um']:.2f} µm"
           f"   ·   {len(evidence['voids'])} void(s) in the controlling cluster",
           fill=(70, 84, 96), font=font(15))
    d.text((PAD, y + 50),
           f"{um:.3f} µm/pixel   ·   void area "
           f"{(mask == VOID_CLASS).sum() * um * um:.0f} µm²   ·   "
           f"yellow ring marks the cluster that set the score",
           fill=(120, 132, 142), font=font(14))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "test_panels"))
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--min-size", type=int, default=MIN_SIZE)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(Path(args.runs).glob("unet_f*.pt"))
    if len(ckpts) != 5:
        sys.exit(f"expected 5 fold checkpoints in {args.runs}, found {len(ckpts)}")
    nets = load_nets([str(c) for c in ckpts], device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    submitted = REPO / "predicted_masks"

    df = index_test().sort_values("stem").reset_index(drop=True)
    print(f"\n{'stem':34} {'severity':>10} {'verdict':>8}  matches submitted")
    fails = 0
    for _, row in df.iterrows():
        stem, um = row["stem"], row["um_per_pixel"]
        img = load_image(row["image"])
        p_void, base = void_prob(nets, img, device)
        mask = to_mask(p_void, base, args.threshold, args.min_size)

        # The whole claim of this folder is "this is the submission". Prove it.
        ref = np.array(Image.open(submitted / f"{stem}.png"))
        assert np.array_equal(mask, ref), f"{stem}: differs from predicted_masks/"

        try:
            severity, _ = compute_max_severity(mask, um)
        except TooManyRegionsError:
            severity = float("inf")
        evidence = controlling_geometry(mask, um)
        fails += severity >= SEVERITY_THRESHOLD

        panel(img, mask, evidence, stem, um, severity).save(out / f"{stem}.png")
        print(f"{stem:34} {severity:10.2f} "
              f"{'FAIL' if severity >= SEVERITY_THRESHOLD else 'PASS':>8}  yes")

    print(f"\n{len(df)} panels -> {out}")
    print(f"{fails} specimens over the {SEVERITY_THRESHOLD} µm limit, {len(df) - fails} under")
    print("every mask verified byte-identical to predicted_masks/")


if __name__ == "__main__":
    main()
