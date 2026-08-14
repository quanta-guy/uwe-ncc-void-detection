"""Score solution 4 exactly the way solution 1's 0.8869 was scored.

    python solution4/evaluate4.py --oof
    python solution4/evaluate4.py --oof --runs runs --pattern "unet_f{f}.pt"   # solution 1

The comparison is the whole point, so the protocol is copied rather than
improved: the same out-of-fold sweep over the same threshold grid, on the same
striding split, including the stored _aug_ duplicates in validation.

That protocol is optimistic - it selects threshold and min_size on the data it
reports, which solution 3 fixed with nested selection. It is used here anyway
because BOTH sides are equally optimistic, so the difference between them is
still a fair read on what the RNG fix bought. Mixing protocols would confound
the one variable this experiment exists to isolate.

Run it twice, once per --runs/--pattern, and compare.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data import VAL_EVERY, index_training  # noqa: E402
from predict import _collect, _sweep, load_nets  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
# The grid src/predict.py --oof used to produce 0.8869.
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
MIN_SIZES = [2, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--pattern", default="s4_unet_f{f}_s0.pt")
    ap.add_argument("--oof", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None, help="Name for the output folder")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cached = ([], [], [], [], [])
    per_fold = []

    for f in range(VAL_EVERY):
        path = Path(args.runs) / args.pattern.format(f=f)
        if not path.exists():
            sys.exit(f"missing {path}")
        nets = load_nets([str(path)], device)
        ckpt = torch.load(path, map_location="cpu")
        assert ckpt.get("fold", f) == f, f"{path.name} says fold {ckpt.get('fold')}"
        df = index_training(f).query("is_val")
        if args.limit:
            df = df.head(args.limit)
        print(f"  fold {f}: {len(df)} held-out images")
        _collect(nets, df, device, tta=False, into=cached)
        per_fold.append(ckpt.get("val_dice", float("nan")))

    print(f"\nout-of-fold over {len(cached[0])} images, all 28 micrographs")
    print(f"per-fold val_dice at training time: "
          f"{', '.join(f'{d:.4f}' for d in per_fold)}")

    best, rows = _sweep(*cached, THRESHOLDS, MIN_SIZES, dilations=[0])
    top = max(rows, key=lambda r: r["final"])

    tag = args.tag or Path(args.pattern.format(f=0)).stem
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "oof_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "best.txt").write_text(
        f"models           {args.runs}/{args.pattern}\n"
        f"per-fold val_dice {', '.join(f'{d:.4f}' for d in per_fold)}\n"
        f"protocol         non-nested OOF, same grid as solution 1's 0.8869\n"
        f"best setting     --threshold {best[1]} --min-size {best[2]}\n"
        f"Dice_void        {top['dice']}\nF2               {top['f2']}\n"
        f"final score      {best[0]:.4f}\n"
        f"TP/FP/FN         {top['tp']}/{top['fp']}/{top['fn']}\n")

    print(f"\n{'=' * 64}")
    print(f"{tag}  OOF final {best[0]:.4f}   Dice {top['dice']:.4f}   "
          f"F2 {top['f2']:.4f}   TP/FP/FN {top['tp']}/{top['fp']}/{top['fn']}")
    print(f"solution 1 reported 0.8869 under this identical protocol")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
