"""Solution 3 training: same model and loss as solution 1, rebuilt data recipe.

    python solution3/train3.py --epochs 30 --fold 0
    python solution3/train3.py --demo

The model is deliberately unchanged - the scratch U-Net, weighted cross-entropy
plus soft Dice on the void channel. Ten architectures were compared on this
data and spanned 0.0245 Dice while folds spanned 0.15; the model was never the
binding constraint. What changes here is everything around it:

  - every image resampled to 0.57 um/pixel, so the network sees one physical
    scale instead of a 2-36px fibre-radius range
  - folds balanced on tiles, void and failing prevalence
  - validation on ORIGINAL images only, no stored _aug_ duplicates
  - augmentation RNG that actually advances between epochs
  - sampling that equalises micrographs and over-samples void-containing tiles

Because sampling is now weighted with replacement, an "epoch" is a fixed number
of draws rather than a pass over the index, and --epochs is not comparable to
solution 1's.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import VOID_CLASS  # noqa: E402
from data3 import (CANONICAL_UM_PER_PX, Micrographs3, balanced_folds,  # noqa: E402
                   balanced_sampler, index3, void_index)
from model import build  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"


def void_dice_loss(logits, target, eps=1.0):
    """Soft Dice on the void channel. Unchanged from solution 1 on purpose."""
    prob = logits.softmax(dim=1)[:, VOID_CLASS]
    tgt = (target == VOID_CLASS).float()
    num = 2 * (prob * tgt).sum(dim=(1, 2)) + eps
    den = prob.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2)) + eps
    return 1 - (num / den).mean()


@torch.no_grad()
def validate(net, loader, device):
    """Mean void Dice over validation images that contain a void.

    Validation runs on whole resampled images, one at a time. Resampling
    produces arbitrary sizes - 149px, 313px - so they cannot be batched
    together, and the U-Net halves resolution four times so each is
    reflect-padded to a multiple of 16 and the logits cropped back. Validating
    on crops instead would measure something the model is never asked to do at
    inference.

    The set is originals only, so each case counts once. Solution 1 counted
    every case twice, once as the original and once as its stored _aug_ copy.
    """
    net.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        h, w = x.shape[-2:]
        ph, pw = (-h) % 16, (-w) % 16
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = net(x)[..., :h, :w].argmax(dim=1)
        p, t = pred == VOID_CLASS, y == VOID_CLASS
        for i in range(len(x)):
            if t[i].any():
                inter = (p[i] & t[i]).sum().item()
                scores.append(2 * inter / (p[i].sum().item() + t[i].sum().item()))
    net.train()
    return float(np.mean(scores)) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200, help="Weighted draws per epoch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--void-boost", type=float, default=3.0)
    ap.add_argument("--val-every", type=int, default=3,
                    help="Validate every N epochs; single-image validation is not free")
    ap.add_argument("--target-um", type=float, default=CANONICAL_UM_PER_PX)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--require-cuda", action="store_true",
                    help="Exit rather than fall back to CPU. On a GPU box a silent "
                         "CPU fallback wastes hours before anyone notices.")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        return demo()

    out = Path(args.out or OUT / f"s3_unet_f{args.fold}_s{args.seed}.pt")
    if args.require_cuda and not torch.cuda.is_available():
        sys.exit("--require-cuda given but torch.cuda.is_available() is False. "
                 f"torch {torch.__version__} - most likely a CPU-only wheel.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    else:
        print(f"device: CPU (torch {torch.__version__}) - training will be very slow")
    torch.manual_seed(args.seed)

    assign, folds = balanced_folds(index3(0))
    df = void_index(index3(max(args.fold, 0), assignment=assign))

    if args.fold < 0:
        # Submission mode: every micrograph trains, nothing is held out. Only
        # for the final ensemble, after the honest number has been measured on
        # the folds - there is no validation signal here, so the epoch count
        # must come from the fold runs rather than from watching val_dice.
        tr, va = df, df[df.is_val].head(0)
        print(f"ALL-DATA seed {args.seed}: {len(tr)} train rows, no validation "
              f"(submission ensemble member)")
    else:
        tr, va = df[df.is_train], df[df.is_val]
        print(f"fold {args.fold} seed {args.seed}: {len(tr)} train rows / "
              f"{len(va)} val originals")
    if args.limit:
        tr, va = tr.head(args.limit), va.head(args.limit)
    print(f"  micrographs {tr.group.nunique()} train / {va.group.nunique()} val, "
          f"canonical {args.target_um} um/px, void boost {args.void_boost}x")

    ds_tr = Micrographs3(tr, train=True, seed=args.seed, target=args.target_um)
    ds_va = Micrographs3(va, train=False, target=args.target_um)
    sampler = balanced_sampler(tr, args.void_boost, args.seed)
    common = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,
                              drop_last=True, **common)
    # batch_size 1: resampling gives every image its own size, so they cannot
    # be collated together.
    val_loader = DataLoader(ds_va, batch_size=1, **common)

    net, _ = build("unet", args.base, args.depth, chroma=False)
    net = net.to(device)
    print(f"  unet base={args.base} depth={args.depth}  "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = min(args.steps, len(train_loader))
    sched = CosineAnnealingLR(opt, T_max=args.epochs * steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    ce_weight = torch.tensor([1.0, 1.0, 5.0], device=device)

    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_loader:
            if n >= steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = net(x)
                loss = F.cross_entropy(logits, y, weight=ce_weight) + void_dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            n += 1

        # With no held-out fold there is nothing to select on, so the last
        # epoch is the checkpoint - picking a "best" would be selecting on
        # training data.
        # Validation is single-image on full resampled frames, so it is not
        # free. Running it every epoch on a 30-epoch schedule costs more than
        # it informs; --val-every 3 keeps the best-checkpoint signal while
        # returning the time to training.
        due = (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1
        dice = validate(net, val_loader, device) if (len(va) and due) else float("nan")
        if not due and len(va):
            print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {total / max(n, 1):.4f}"
                  f"  {time.time() - t0:5.1f}s")
            continue
        flag = ""
        if len(va) == 0:
            best = dice
            torch.save({"model": net.state_dict(), "arch": "unet", "base": args.base,
                        "depth": args.depth, "chroma": False, "fold": args.fold,
                        "seed": args.seed, "val_dice": float("nan"),
                        "pipeline": "solution3", "target_um": args.target_um,
                        "void_boost": args.void_boost, "fold_assignment": assign}, out)
        elif dice > best:
            best = dice
            torch.save({"model": net.state_dict(), "arch": "unet", "base": args.base,
                        "depth": args.depth, "chroma": False, "fold": args.fold,
                        "seed": args.seed, "val_dice": dice, "pipeline": "solution3",
                        "target_um": args.target_um, "void_boost": args.void_boost,
                        "fold_assignment": assign}, out)
            flag = "  *"
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {total / max(n, 1):.4f}  "
              f"val_dice {dice:.4f}  {time.time() - t0:5.1f}s{flag}")

    print(f"\nbest val_dice {best:.4f} -> {out}")


def demo():
    """Loss falls, and the pieces this solution depends on actually hold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = build("unet", base=8, depth=2)
    net = net.to(device).train()

    x = torch.randn(2, 3, 64, 64, device=device)
    y = torch.zeros(2, 64, 64, dtype=torch.long, device=device)
    y[:, 20:40, 20:40] = VOID_CLASS

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        logits = net(x)
        loss = F.cross_entropy(logits, y) + void_dice_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"

    empty = torch.zeros(1, 3, 8, 8)
    empty[:, 0] = 20.0
    tgt = torch.zeros(1, 8, 8, dtype=torch.long)
    tgt[:, 2:5, 2:5] = VOID_CLASS
    assert void_dice_loss(empty, tgt).item() > 0.8, "all-background escaped the Dice term"

    # The sampler must actually raise the void share, or foreground
    # oversampling is silently a no-op.
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 90 + ["b"] * 10,
                       "has_void": [False] * 95 + [True] * 5})
    s = balanced_sampler(df, void_boost=3.0, seed=0)
    drawn = list(s)
    void_share = np.mean([df.iloc[i]["has_void"] for i in drawn])
    assert void_share > 0.05, f"void share not boosted: {void_share:.3f}"
    # Micrograph 'b' has 10 rows against 'a's 90 but must be sampled comparably.
    b_share = np.mean([df.iloc[i]["group"] == "b" for i in drawn])
    assert b_share > 0.25, f"small micrograph still under-sampled: {b_share:.3f}"

    print(f"ok  loss {losses[0]:.3f} -> {losses[-1]:.3f}  "
          f"sampler void share {void_share:.1%}, small-micrograph share {b_share:.1%}")


if __name__ == "__main__":
    main()
