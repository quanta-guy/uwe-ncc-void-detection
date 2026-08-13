"""Evaluate every trained checkpoint, one results directory per model.

    python src/bench.py                    # every runs/*.pt, plus the fold ensemble
    python src/bench.py --ckpt runs/arch_unetpp_r34.pt

Each run gets results/<run_id>/, and each model a folder inside it:

    sweep.csv              the full threshold x min_size x dilate grid
    best.txt               the winning setting and its score
    panels/                overlay panels, worst disagreements first
    inspection_queue.csv   the accept/review/reject report on the Test set

plus run.json (host, GPU, git commit, grid) and comparison.csv across
all models. Runs are never overwritten - a later run with different
checkpoints or a different grid is a different folder.

Training happens wherever there is a GPU; this runs on the laptop, because
scoring is CPU-bound severity geometry rather than matrix multiplication, and
because the comparison is worth keeping next to the code you are writing
about it.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
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
LOWS = [None, 0.05, 0.1]
GACS = [0, 5, 15]


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
    ap.add_argument("--run-id", default=None,
                    help="Folder name under results/; defaults to a UTC timestamp")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = [(Path(c).stem, [Path(c)]) for c in args.ckpt] if args.ckpt else discover(args.runs)
    if not jobs:
        sys.exit(f"no checkpoints in {args.runs}")

    # Every evaluation gets its own folder. Runs differ by checkpoint set, grid
    # and code version, so overwriting one with the next loses the comparison
    # that made it worth running.
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out) / run_id
    root.mkdir(parents=True, exist_ok=True)

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"

    meta = {
        "run_id": run_id,
        "started_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "host": platform.node(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "grid": {"thresholds": THRESHOLDS, "min_sizes": MIN_SIZES,
                 "dilations": DILATIONS, "lows": LOWS, "gacs": GACS},
        "val_images_per_model": args.limit or "all",
        "models": [n for n, _ in jobs],
    }
    (root / "run.json").write_text(json.dumps(meta, indent=2))

    print(f"run {run_id}: {len(jobs)} model(s) on {device}")
    print(f"  -> {root}\n")
    summary = []

    for name, ckpts in jobs:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        out = root / name
        (out / "panels").mkdir(parents=True, exist_ok=True)

        info = describe(ckpts[0])
        nets = load_nets([str(c) for c in ckpts], device)

        # Fold models must be graded on the fold they held out, or the score is
        # a model marking its own homework. Anything else uses fold 0.
        fold = info["fold"] if len(ckpts) == 1 else 0
        df = index_training(fold).query("is_val")
        if args.limit:
            df = df.head(args.limit)

        best, rows = tune(nets, df, device, THRESHOLDS, MIN_SIZES, DILATIONS, LOWS, GACS)
        final, t, m, d, lo, g = best

        with open(out / "sweep.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        top = max(rows, key=lambda r: r["final"])
        (out / "best.txt").write_text(
            f"model            {name}\n"
            f"checkpoints      {', '.join(c.name for c in ckpts)}\n"
            f"arch             {info['arch']}  depth {info['depth']}  base {info['base']}\n"
            f"graded on fold   {fold}  ({len(df)} held-out images)\n"
            f"best setting     --threshold {t} --min-size {m} --dilate {d}\n"
            f"Dice_void        {top['dice']}\n"
            f"F2               {top['f2']}\n"
            f"final score      {final:.4f}\n"
            f"TP/FP/FN         {top['tp']}/{top['fp']}/{top['fn']}\n")

        summary.append({"model": name, **info, "graded_on_fold": fold,
                        "threshold": t, "min_size": m, "dilate": d, "low": lo, "gac": g,
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
    comp = root / "comparison.csv"
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

    (root / "run.json").write_text(json.dumps(
        {**meta,
         "finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
         "dice_spread": round(spread, 4),
         "ranking": [{k: r[k] for k in ("model", "arch", "dice", "f2", "final")}
                     for r in summary]},
        indent=2))

    print(f"\nrun {run_id}\nper-model directories: {root}\ncomparison: {comp}")


if __name__ == "__main__":
    main()
