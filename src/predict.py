"""Predict masks, and tune the two decision knobs against the real score.

    python src/predict.py --tune          # pick threshold + min_size on val
    python src/predict.py                 # write predicted_masks/ for the Test set

Two knobs sit between the network and the score, and neither is learned:

  --threshold  a pixel is void when P(void) exceeds this, instead of when void
               merely wins the argmax. F2 weighs a missed failure 4x a false
               alarm, so the score-optimal threshold is below 0.5.
  --min-size   void blobs smaller than this are deleted. Speckle inflates the
               severity sum (and past 1500 regions is scored as an automatic
               fail), so this is a correctness guard, not a cosmetic one.

--tune scores a grid of both on the held-out split using evaluation.py's own
functions, so the number it reports is the challenge's number, not a proxy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.morphology import binary_dilation, disk, remove_small_objects

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data import VAL_EVERY, VOID_CLASS, index_test, index_training, load_image, load_mask, to_tensor  # noqa: E402
from evaluation import (  # noqa: E402  - the judge's own scoring code, reused as-is
    SEVERITY_THRESHOLD,
    TooManyRegionsError,
    compute_f2,
    compute_max_severity,
    dice_void,
)
from model import build  # noqa: E402

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SIZE = 10
DEFAULT_DILATE = 0


def load_nets(ckpt_paths, device):
    """One or more checkpoints. Several are averaged as an ensemble.

    Runs that differ only in LR schedule still land in different minima, so
    averaging them buys real diversity - unlike TTA, which only averages
    symmetries the model was already trained to be invariant to.
    """
    nets = []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location=device)
        net, _ = build(ckpt.get("arch", "unet"), ckpt.get("base", 32), ckpt.get("depth", 4))
        net = net.to(device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        nets.append(net)
        print(f"  {path}  (val_dice {ckpt.get('val_dice', float('nan')):.4f})")
    return nets


def _dihedral_ops(square):
    """(rotations, flip) pairs covering the symmetries training saw.

    Training augments over the full dihedral group, so averaging predictions
    over it at inference is the matched test-time transform. A quarter turn
    swaps height and width, so non-square inputs get the shape-preserving
    subgroup only.
    """
    turns = range(4) if square else (0, 2)
    return [(k, flip) for k in turns for flip in (False, True)]


@torch.no_grad()
def void_prob(nets, img, device, tta=False):
    """P(void) per pixel, plus the matrix/fibre call for everywhere else."""
    x = to_tensor(img)[None].to(device)

    # The U-Net halves resolution 4 times, so pad up to a multiple of 16 and
    # crop the padding back off. Test images need not be 256x256.
    h, w = img.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")

    ops = _dihedral_ops(x.shape[-2] == x.shape[-1]) if tta else [(0, False)]

    # All variants go through as one batch - 8 images of this size is nothing,
    # and it keeps TTA to a single kernel launch per layer.
    batch = torch.cat([
        torch.rot90(torch.flip(x, [-1]) if f else x, k, (-2, -1)) for k, f in ops
    ])

    with torch.autocast(device.type, enabled=device.type == "cuda"):
        out = torch.stack([n(batch).float().softmax(dim=1) for n in nets]).mean(0)

    # Undo each transform in reverse order before averaging.
    prob = torch.stack([
        torch.flip(u, [-1]) if f else u
        for (k, f), o in zip(ops, out)
        for u in [torch.rot90(o[None], -k, (-2, -1))]
    ]).mean(0)[0, :, :h, :w]

    return prob[VOID_CLASS].cpu().numpy(), prob[:VOID_CLASS].argmax(dim=0).cpu().numpy().astype(np.uint8)


def to_mask(p_void, base, threshold=DEFAULT_THRESHOLD, min_size=DEFAULT_MIN_SIZE,
            dilate=DEFAULT_DILATE):
    """Threshold, despeckle, optionally dilate, stamp over the base labels.

    `dilate` corrects a measured bias: the model recovers only 10-25% of the
    void pixels on the specimens it misses, so the severity computed from our
    masks reads far lower than the same void's severity in the ground truth -
    and the fail line of 25 was calibrated on ground-truth masks. Growing the
    void regions puts both on the same ruler. It cannot be done at scoring
    time, because judging applies its own line of 25 to whatever mask we
    submit; the correction has to live in the mask itself.

    Despeckling happens first, so dilation grows real voids and not noise.
    """
    void = p_void > threshold
    if min_size > 1:
        void = remove_small_objects(void, min_size=min_size)
    if dilate:
        void = binary_dilation(void, disk(dilate))
    mask = base.copy()
    mask[void] = VOID_CLASS
    return mask


def write_masks(nets, df, out_dir, device, threshold, min_size, dilate=0, tta=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        p_void, base = void_prob(nets, load_image(row["image"]), device, tta)
        mask = to_mask(p_void, base, threshold, min_size, dilate)
        Image.fromarray(mask).save(out_dir / f"{row['stem']}.png")
    print(f"wrote {len(df)} masks to {out_dir}")


# -----------------------------
# TUNING
# -----------------------------
def gt_facts(gts, um_per_px):
    """(severity, n_void_regions) per ground-truth mask.

    Independent of any knob setting, so it is computed once and reused across
    the whole sweep rather than recomputed per grid point.
    """
    return [compute_max_severity(gt, um) for gt, um in zip(gts, um_per_px)]


def _measure(masks, gts, um_per_px, facts):
    """Dice mean and per-image predicted severity for one knob setting."""
    dices, sevs = [], []

    for mask, gt, um, (_, gt_voids) in zip(masks, gts, um_per_px, facts):
        if gt_voids > 0:
            dices.append(dice_void(gt, mask))
        try:
            sev, _ = compute_max_severity(mask, um)
        except TooManyRegionsError:
            sev = float("inf")  # judging scores unscorable predictions as a fail
        sevs.append(sev)

    return (float(np.mean(dices)) if dices else float("nan")), dices, sevs


def _score_at(sevs, facts, mean_dice, has_dice):
    """Score at the real fail line, which judging applies to both sides."""
    tp = fp = fn = 0
    for sev, (gt_sev, _) in zip(sevs, facts):
        gt_fail = gt_sev >= SEVERITY_THRESHOLD
        pred_fail = sev >= SEVERITY_THRESHOLD
        if gt_fail and pred_fail:
            tp += 1
        elif gt_fail:
            fn += 1
        elif pred_fail:
            fp += 1

    f2 = compute_f2(tp, fp, fn)
    gate = min(1, mean_dice / 0.8) if has_dice else 1.0
    return f2 * gate, f2, tp, fp, fn


def _collect(nets, df, device, tta, into=None):
    """Cache void probabilities and ground truth for a set of images."""
    probs, bases, gts, ums = into if into else ([], [], [], [])
    for _, row in df.iterrows():
        p_void, base = void_prob(nets, load_image(row["image"]), device, tta)
        probs.append(p_void.astype(np.float16))
        bases.append(base)
        gts.append(load_mask(row["mask"]))
        ums.append(row["um_per_pixel"])
    return probs, bases, gts, ums


def oof(ckpt_dir, device, thresholds, min_sizes, dilations, tta=False):
    """Out-of-fold sweep: every image scored by the one model that never saw it.

    The single-split number rests on 6 micrographs and is noisy enough that
    best-epoch selection is partly luck. This covers all 28, once each, with
    no model ever grading its own training data.
    """
    cached = ([], [], [], [])
    for f in range(VAL_EVERY):
        nets = load_nets([str(Path(ckpt_dir) / f"unet_f{f}.pt")], device)
        df = index_training(f).query("is_val")
        print(f"  fold {f}: {len(df)} held-out images")
        _collect(nets, df, device, tta, into=cached)
    return _sweep(*cached, thresholds, min_sizes, dilations)


def tune(nets, df, device, thresholds, min_sizes, dilations, tta=False):
    print(f"caching predictions for {len(df)} val images{' (TTA x8)' if tta else ''}...")
    probs, bases, gts, ums = _collect(nets, df, device, tta)
    return _sweep(probs, bases, gts, ums, thresholds, min_sizes, dilations)


def _sweep(probs, bases, gts, ums, thresholds, min_sizes, dilations):
    facts = gt_facts(gts, ums)
    print(f"ground truth: {sum(1 for _, n in facts if n)} void-containing, "
          f"{sum(1 for s, _ in facts if s >= SEVERITY_THRESHOLD)} failing")

    print(f"\n{'thresh':>7} {'min_size':>9} {'dilate':>7} {'Dice':>7} {'F2':>7} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'FINAL':>7}")
    best, rows = None, []
    for t in thresholds:
        for m in min_sizes:
            for d in dilations:
                masks = [to_mask(p.astype(np.float32), b, t, m, d) for p, b in zip(probs, bases)]
                mean_dice, dices, sevs = _measure(masks, gts, ums, facts)
                final, f2, tp, fp, fn = _score_at(sevs, facts, mean_dice, bool(dices))
                print(f"{t:7.2f} {m:9d} {d:7d} {mean_dice:7.4f} {f2:7.4f} "
                      f"{tp:4d} {fp:4d} {fn:4d} {final:7.4f}")
                rows.append({"threshold": t, "min_size": m, "dilate": d,
                             "dice": round(mean_dice, 4), "f2": round(f2, 4),
                             "tp": tp, "fp": fp, "fn": fn, "final": round(final, 4)})
                if best is None or final > best[0]:
                    best = (final, t, m, d)

    print(f"\nbest: --threshold {best[1]} --min-size {best[2]} --dilate {best[3]}"
          f"  (final score {best[0]:.4f})")
    return best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[str(REPO / "runs" / "unet.pt")],
                    help="One checkpoint, or several to average as an ensemble")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--out", default=str(REPO / "predicted_masks"))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    ap.add_argument("--dilate", type=int, default=DEFAULT_DILATE,
                    help="Grow predicted voids by this radius, in pixels")
    ap.add_argument("--tune", action="store_true", help="Sweep threshold/min-size on the val split")
    ap.add_argument("--oof", action="store_true",
                    help="Sweep out-of-fold across all 5 folds (needs runs/unet_f0..4.pt)")
    ap.add_argument("--tta", action="store_true", help="Average predictions over the 8 dihedral transforms")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="Run the self-check and exit")
    args = ap.parse_args()

    if args.demo:
        return demo()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.oof:
        oof(REPO / "runs", device,
            thresholds=[0.2, 0.3, 0.4, 0.5],
            min_sizes=[2, 4],
            dilations=[0],
            tta=args.tta)
        return

    nets = load_nets(args.ckpt, device)

    if args.tune:
        df = index_training()
        df = df[df.is_val]
        if args.limit:
            df = df.head(args.limit)
        tune(nets, df, device,
             thresholds=[0.2, 0.3, 0.4, 0.5],
             min_sizes=[2, 4],
             dilations=[0, 1, 2, 3],
             tta=args.tta)
        return

    if args.split == "test":
        df = index_test()
    else:
        df = index_training()
        df = df[df.is_val]
    if args.limit:
        df = df.head(args.limit)

    write_masks(nets, df, args.out, device, args.threshold, args.min_size,
                args.dilate, args.tta)


def demo():
    """to_mask honours both knobs and never invents a class outside 0/1/2."""
    base = np.zeros((32, 32), dtype=np.uint8)
    base[16:, :] = 1
    p = np.zeros((32, 32), dtype=np.float32)
    p[2:8, 2:8] = 0.9   # 36 px blob
    p[20, 20] = 0.9     # 1 px speckle
    p[10:12, 10:12] = 0.35  # only survives a low threshold

    m = to_mask(p, base, threshold=0.5, min_size=10)
    assert (m == VOID_CLASS).sum() == 36, (m == VOID_CLASS).sum()  # speckle removed
    assert set(np.unique(m)) <= {0, 1, 2}
    assert m[20, 0] == 1 and m[0, 0] == 0  # base labels survive underneath

    m = to_mask(p, base, threshold=0.3, min_size=1)
    assert (m == VOID_CLASS).sum() == 36 + 1 + 4

    # Dilation must grow the surviving blob and not resurrect the despeckled
    # one - despeckling has to happen first or noise gets amplified.
    grown = to_mask(p, base, threshold=0.5, min_size=10, dilate=2)
    assert (grown == VOID_CLASS).sum() > 36
    assert grown[18:23, 18:23].max() < VOID_CLASS, "dilation regrew removed speckle"

    # TTA's classic silent failure is un-transforming in the wrong order: the
    # average still looks plausible, it is just built from misaligned maps.
    # Round-tripping every op catches that; nothing downstream would.
    t = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    for k, f in _dihedral_ops(square=False):
        fwd = torch.rot90(torch.flip(t, [-1]) if f else t, k, (-2, -1))
        inv = torch.rot90(fwd, -k, (-2, -1))
        assert torch.equal(torch.flip(inv, [-1]) if f else inv, t), (k, f)
    assert len(_dihedral_ops(square=True)) == 8
    assert len(_dihedral_ops(square=False)) == 4  # quarter turns would reshape
    print("ok")


if __name__ == "__main__":
    main()
