"""Evaluate every trained checkpoint, one results directory per model.

    python src/bench.py                    # every runs/*.pt, plus the fold ensemble
    python src/bench.py --ckpt runs/arch_unetpp_r34.pt

Each model gets results/<name>/ containing:

    sweep.csv              the full threshold x min_size x dilate grid
    best.txt               the winning setting and its score
    panels/                overlay panels, worst disagreements first
    inspection_queue.csv   the accept/review/reject report on the Test set

and every model lands one row in results/comparison.csv.

Training happens wherever there is a GPU; this runs on the laptop, because
scoring is CPU-bound severity geometry rather than matrix multiplication, and
because the comparison is worth keeping next to the code you are writing
about it.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data import index_training  # noqa: E402
from predict import load_nets, tune  # noqa: E402

SRC = REPO / "src"
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]
MIN_SIZES = [2, 4, 10]
DILATIONS = [0]


def discover(runs_dir):
    """Every checkpoint, plus the fold ensemble if all five folds exist."""
    jobs = [(p.stem, [p]) for p in sorted(Path(runs_dir).glob("*.pt"))]
    folds = [Path(runs_dir) / f"unet_f{i}.pt" for i in range(5)]
    if all(f.exists() for f in folds):
        jobs.append(("ensemble_5fold", folds))
    return jobs


def describe(ckpt):
    c = torch.load(ckpt, map_location="cpu")
    return {"arch": c.get("arch", "unet"), "depth": c.get("depth", 4),
            "base": c.get("base", 32), "fold": c.get("fold", 0),
            "train_val_dice": round(c.get("val_dice", float("nan")), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", help="Specific checkpoints; default is all of runs/")
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--panels", type=int, default=12, help="Overlay panels per model, 0 to skip")
    ap.add_argument("--limit", type=int, default=0, help="Val images per model, for a quick pass")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = [(Path(c).stem, [Path(c)]) for c in args.ckpt] if args.ckpt else discover(args.runs)
    if not jobs:
        sys.exit(f"no checkpoints in {args.runs}")

    print(f"{len(jobs)} model(s) to evaluate on {device}\n")
    summary = []

    for name, ckpts in jobs:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        out = Path(args.out) / name
        (out / "panels").mkdir(parents=True, exist_ok=True)

        meta = describe(ckpts[0])
        nets = load_nets([str(c) for c in ckpts], device)

        # Fold models must be graded on the fold they held out, or the score is
        # a model marking its own homework. Anything else uses fold 0.
        fold = meta["fold"] if len(ckpts) == 1 else 0
        df = index_training(fold).query("is_val")
        if args.limit:
            df = df.head(args.limit)

        best, rows = tune(nets, df, device, THRESHOLDS, MIN_SIZES, DILATIONS)
        final, t, m, d = best

        with open(out / "sweep.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        top = max(rows, key=lambda r: r["final"])
        (out / "best.txt").write_text(
            f"model            {name}\n"
            f"checkpoints      {', '.join(c.name for c in ckpts)}\n"
            f"arch             {meta['arch']}  depth {meta['depth']}  base {meta['base']}\n"
            f"graded on fold   {fold}  ({len(df)} held-out images)\n"
            f"best setting     --threshold {t} --min-size {m} --dilate {d}\n"
            f"Dice_void        {top['dice']}\n"
            f"F2               {top['f2']}\n"
            f"final score      {final:.4f}\n"
            f"TP/FP/FN         {top['tp']}/{top['fp']}/{top['fn']}\n")

        summary.append({"model": name, **meta, "graded_on_fold": fold,
                        "threshold": t, "min_size": m, "dilate": d,
                        "dice": top["dice"], "f2": top["f2"], "final": round(final, 4),
                        "tp": top["tp"], "fp": top["fp"], "fn": top["fn"]})

        if args.panels:
            ck = [str(c) for c in ckpts]
            for script, sub, extra in [("overlay.py", "panels", ["--split", "val",
                                                                 "--limit", str(args.panels)]),
                                       ("report.py", "", ["--split", "test"])]:
                dest = out / sub if sub else out
                subprocess.run([sys.executable, str(SRC / script), "--ckpt", *ck,
                                "--threshold", str(t), "--min-size", str(m),
                                "--out", str(dest), *extra],
                               cwd=REPO, check=False)
            # overlay.py nests by split; flatten so the folder is just panels.
            nested = out / "panels" / "val"
            if nested.exists():
                for f in nested.glob("*.png"):
                    f.rename(out / "panels" / f.name)
                nested.rmdir()

    summary.sort(key=lambda r: -r["final"])
    comp = Path(args.out) / "comparison.csv"
    with open(comp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print(f"\n\n{'=' * 78}\nCOMPARISON\n{'=' * 78}")
    print(f"{'model':22} {'arch':14} {'fold':>4} {'Dice':>7} {'F2':>7} {'final':>7}")
    for r in summary:
        print(f"{r['model'][:22]:22} {r['arch'][:14]:14} {r['graded_on_fold']:4d} "
              f"{r['dice']:7.4f} {r['f2']:7.4f} {r['final']:7.4f}")

    spread = summary[0]["dice"] - summary[-1]["dice"]
    print(f"\nDice spread across {len(summary)} models: {spread:.4f}")
    print("A small spread across architectures points at the labels, not the model.")
    print(f"\nper-model directories: {Path(args.out)}\ncomparison: {comp}")


if __name__ == "__main__":
    main()
