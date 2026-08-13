"""Train the void segmentation U-Net.

    python src/train.py --epochs 30

The validation number printed each epoch is mean Dice on the void class over
void-containing val images only - the same quantity the challenge's Dice gate
grades, so it is directly comparable to the 0.8 the gate saturates at.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import VOID_CLASS, Micrographs, index_training
from model import SMP_ARCHS, build

REPO = Path(__file__).resolve().parents[1]


def void_dice_loss(logits, target, eps=1.0):
    """Soft Dice on the void channel.

    Void pixels are a small fraction of the image, so plain cross-entropy is
    happy to predict "no void" everywhere. Dice is computed on the class the
    score actually cares about, which pulls against that directly.
    """
    prob = logits.softmax(dim=1)[:, VOID_CLASS]
    tgt = (target == VOID_CLASS).float()
    num = 2 * (prob * tgt).sum(dim=(1, 2)) + eps
    den = prob.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2)) + eps
    return 1 - (num / den).mean()


def batch_void_dice(pred, target):
    """Hard per-image Dice on void, and whether the image has any void in GT.

    Mirrors evaluation.py's dice_void, including its convention that two empty
    masks score 1.0 - the caller drops void-free images so that convention
    cannot inflate the mean.
    """
    p = (pred == VOID_CLASS)
    t = (target == VOID_CLASS)
    inter = (p & t).sum(dim=(1, 2)).float()
    total = p.sum(dim=(1, 2)).float() + t.sum(dim=(1, 2)).float()
    dice = torch.where(total == 0, torch.ones_like(total), 2 * inter / total.clamp(min=1))
    return dice, t.any(dim=(1, 2))


@torch.no_grad()
def validate(net, loader, device):
    net.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = net(x).argmax(dim=1)
        dice, has_void = batch_void_dice(pred, y)
        scores.append(dice[has_void].cpu())
    scores = torch.cat(scores) if scores else torch.zeros(0)
    return scores.mean().item() if len(scores) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arch", default="unet", choices=["unet", *SMP_ARCHS],
                    help="Scratch U-Net, or an ImageNet-pretrained variant")
    ap.add_argument("--base", type=int, default=32, help="U-Net width")
    ap.add_argument("--depth", type=int, default=4,
                    help="Downsampling stages. At 4, a median 15px void is "
                         "sub-pixel at the bottleneck; 3 keeps twice the detail")
    ap.add_argument("--out", default=str(REPO / "runs" / "unet.pt"))
    ap.add_argument("--fold", type=int, default=0, help="Which micrograph slice to hold out (0-4)")
    ap.add_argument("--limit", type=int, default=0, help="Smoke test: use N training images")
    ap.add_argument("--demo", action="store_true", help="Run the self-check and exit")
    args = ap.parse_args()

    if args.demo:
        return demo()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = index_training(args.fold)
    train_df, val_df = df[~df.is_val], df[df.is_val]
    if args.limit:
        train_df, val_df = train_df.head(args.limit), val_df.head(max(4, args.limit // 4))
    print(f"train: {len(train_df)}  val: {len(val_df)}")

    common = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(Micrographs(train_df, train=True), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(Micrographs(val_df), batch_size=args.batch_size, **common)

    net, pretrained = build(args.arch, args.base, args.depth)
    net = net.to(device)
    print(f'arch: {args.arch}  params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M'
          f"{'  (imagenet-normalised)' if pretrained else ''}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(train_loader)))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # ponytail: fixed class weights, not computed from the data. The Dice term
    # is what actually handles the void imbalance; these only stop matrix
    # pixels from dominating early. Compute real frequencies if it plateaus.
    ce_weight = torch.tensor([1.0, 1.0, 5.0], device=device)

    best = -1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        net.train()
        started, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = net(x)
                loss = F.cross_entropy(logits, y, weight=ce_weight) + void_dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()

        dice = validate(net, val_loader, device)
        flag = ""
        if dice > best:
            best = dice
            torch.save({"model": net.state_dict(), "base": args.base,
                        "depth": args.depth, "arch": args.arch, "val_dice": dice,
                        "fold": args.fold}, args.out)
            flag = "  <- saved"
        print(f"epoch {epoch:3d}  loss {running / max(1, len(train_loader)):.4f}  "
              f"val_dice_void {dice:.4f}  {time.time() - started:.0f}s{flag}")

    print(f"\nbest val Dice_void: {best:.4f}  ->  {args.out}")
    print("Next: python src/predict.py --tune")


def demo():
    """Losses and metric agree with their definitions on hand-made cases."""
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, :4, :4] = VOID_CLASS

    perfect = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 20
    assert void_dice_loss(perfect, target) < 0.05
    assert void_dice_loss(-perfect, target) > 0.9

    dice, has_void = batch_void_dice(target, target)
    assert has_void.all() and torch.allclose(dice, torch.ones(1))

    half = target.clone()
    half[0, :2, :4] = 0  # keep half the void pixels -> Dice = 2*8/(8+16) = 2/3
    dice, _ = batch_void_dice(half, target)
    assert abs(dice.item() - 2 / 3) < 1e-6, dice

    empty = torch.zeros(1, 8, 8, dtype=torch.long)
    dice, has_void = batch_void_dice(empty, empty)
    assert dice.item() == 1.0 and not has_void.any()  # dropped from the mean
    print("ok")


if __name__ == "__main__":
    main()
