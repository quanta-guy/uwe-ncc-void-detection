"""Train the scratch U-Net on the albumentations pipeline, 3-class head.

    python solution2/train.py --epochs 20 --fold 0
    python solution2/train.py --demo            # self-check, no data needed

Loss is BCEWithLogitsLoss + smp's Dice, in multilabel mode over three channels.
BCE is per pixel and would rather call everything matrix when voids are well
under 1% of the image; Dice is a ratio and scores an all-background prediction
at zero however rare the class. BCE alone converges to the degenerate answer,
Dice alone gives noisy gradients on images containing no void at all. Together
each covers the other's failure.

pos_weight carries the class imbalance, at 5x on the void channel - the same
correction solution 1 made, and it is not arbitrary: F2 weighs a missed failure
4x a false alarm, so the loss should not be left at whatever the pixel counts
happen to be.

The architecture is the scratch U-Net rather than an smp backbone, for two
reasons that both showed up in measurement. The smp architectures wrap
themselves in _InputAdapter and normalise internally, which would collide with
A.Normalize in the Compose; and every ImageNet encoder tried on this data lost
to the scratch model, because natural-image priors do not transfer to
micrographs.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import N_CLASSES, VOID_CLASS, index_training  # noqa: E402
from micronet import build_micronet  # noqa: E402
from model import build  # noqa: E402
from pipeline import Micrographs2, build_transforms  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"


class BCEDiceLoss(nn.Module):
    """BCEWithLogits + soft Dice over all three class channels.

    Both take raw logits: BCEWithLogitsLoss folds the sigmoid in for numerical
    stability, and smp's DiceLoss does its own when from_logits=True. Applying
    sigmoid beforehand would activate twice and flatten the gradients - a
    silent failure that simply trains badly.
    """

    def __init__(self, void_weight=5.0, dice_weight=1.0):
        super().__init__()
        w = torch.ones(N_CLASSES)
        w[VOID_CLASS] = void_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=w.view(-1, 1, 1))
        self.dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.bce(logits, target) + self.dice_weight * self.dice(logits, target)


@torch.no_grad()
def validate(net, loader, device, threshold=0.5):
    """Mean void Dice over images that actually contain a void.

    Void-free images would score a perfect 1 for predicting nothing, and most
    images are void-free, so averaging them in reports a number that barely
    moves however badly the voids themselves are segmented.
    """
    net.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = net(x).float().sigmoid()[:, VOID_CLASS] > threshold
        t = y[:, VOID_CLASS] > 0.5
        for i in range(len(x)):
            if t[i].any():
                inter = (pred[i] & t[i]).sum().item()
                scores.append(2 * inter / (pred[i].sum().item() + t[i].sum().item()))
    net.train()
    return float(np.mean(scores)) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet",
                    help="'unet' is the scratch model; anything else is an smp class (Unet, FPN, ...)")
    ap.add_argument("--encoder", default="resnet50", help="smp encoder, when --arch is an smp class")
    ap.add_argument("--weights", default="micronet",
                    choices=["micronet", "image-micronet", "imagenet", "none"],
                    help="Encoder pretraining, when --arch is an smp class")
    ap.add_argument("--norm", choices=["single", "imagenet", "micronet"], default=None,
                    help="Defaults to the statistics the encoder was pretrained under")
    ap.add_argument("--aug", choices=["full", "thin"], default="full",
                    help="'thin' is the original resize/flip/rot90/brightness stack")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr-scale", type=float, default=0.1,
                    help="Pretrained encoder lr as a fraction of --lr. 1.0 reproduces "
                         "the single-lr run that showed no benefit from pretraining.")
    ap.add_argument("--void-weight", type=float, default=5.0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Cap images per split, for a smoke test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        return demo()

    # Normalisation follows the encoder unless overridden. Scoring reads it back
    # off the checkpoint, so a model can never be evaluated under statistics it
    # was not trained with - the classic silent accuracy leak here.
    if args.norm is None:
        args.norm = "single" if args.arch == "unet" else {
            "micronet": "micronet", "image-micronet": "micronet"}.get(args.weights, "imagenet")

    tag = "unet" if args.arch == "unet" else f"{args.arch.lower()}_{args.encoder}_{args.weights}"
    out = Path(args.out or OUT / f"alb_{tag}_f{args.fold}.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = index_training(args.fold)
    tr, va = df[~df.is_val], df[df.is_val]
    if args.limit:
        tr, va = tr.head(args.limit), va.head(args.limit)
    print(f"fold {args.fold}: {len(tr)} train / {len(va)} val "
          f"({tr.group.nunique()} / {va.group.nunique()} micrographs) on {device}")

    train_t, val_t = build_transforms(args.norm, aug=args.aug)
    common = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(Micrographs2(tr, train_t), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(Micrographs2(va, val_t), batch_size=args.batch_size, **common)

    if args.arch == "unet":
        # build() returns a bare U-Net with no internal normalisation, so
        # A.Normalize in the Compose is the only one applied.
        net, pretrained = build("unet", args.base, args.depth, chroma=False)
        assert not pretrained, "scratch path must not return a wrapped smp model"
        desc = f"unet(scratch) base={args.base} depth={args.depth}"
    elif args.weights in ("micronet", "image-micronet"):
        net = build_micronet(args.arch, args.encoder, args.weights)
        desc = f"{args.arch}/{args.encoder} <- {args.weights}"
    else:
        # Built directly rather than through build(), which would wrap it in
        # _InputAdapter and normalise a second time on top of the Compose.
        net = getattr(smp, args.arch)(
            encoder_name=args.encoder,
            encoder_weights=None if args.weights == "none" else args.weights,
            in_channels=3, classes=N_CLASSES)
        desc = f"{args.arch}/{args.encoder} <- {args.weights}"

    net = net.to(device)
    print(f"{desc} norm={args.norm} aug={args.aug}  "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")

    criterion = BCEDiceLoss(args.void_weight).to(device)

    # A pretrained encoder must not be updated as hard as a randomly
    # initialised decoder. At one shared learning rate the decoder's early
    # gradients - large, because it starts from noise - are applied to encoder
    # weights that already know something, and the pretraining is destroyed in
    # the first epochs. The result trains fine and scores like a scratch model,
    # which is exactly what MicroNet did at scale 1.0.
    if args.arch != "unet" and args.weights != "none" and args.encoder_lr_scale != 1.0:
        enc = list(net.encoder.parameters())
        enc_ids = {id(p) for p in enc}
        dec = [p for p in net.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW(
            [{"params": enc, "lr": args.lr * args.encoder_lr_scale},
             {"params": dec, "lr": args.lr}], weight_decay=1e-4)
        print(f"  encoder lr {args.lr * args.encoder_lr_scale:.1e} "
              f"({len(enc)} tensors), decoder lr {args.lr:.1e} ({len(dec)} tensors)")
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # CosineAnnealingLR anneals each param group from its own initial lr, so the
    # ratio between encoder and decoder is preserved for the whole schedule.
    sched = CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        t0, total = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()

        dice = validate(net, val_loader, device)
        flag = ""
        if dice > best:
            best = dice
            # Everything evaluate.py needs to rebuild the model and match the
            # preprocessing. Reading these from the checkpoint rather than the
            # command line makes it impossible to score under the wrong norm.
            torch.save({"model": net.state_dict(), "arch": args.arch, "base": args.base,
                        "depth": args.depth, "chroma": False, "norm": args.norm,
                        "encoder": args.encoder, "weights": args.weights,
                        "encoder_lr_scale": args.encoder_lr_scale, "epochs": args.epochs,
                        "aug": args.aug, "fold": args.fold, "val_dice": dice,
                        "classes": N_CLASSES, "pipeline": "albumentations-v2",
                        "size": 256}, out)
            flag = "  *"
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {total / len(train_loader):.4f}  "
              f"val_dice {dice:.4f}  {time.time() - t0:5.1f}s{flag}")

    print(f"\nbest val_dice {best:.4f} -> {out}")


def demo():
    """The loss must fall on a batch small enough to memorise, and must punish
    the degenerate all-background answer that BCE alone tolerates."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = build("unet", base=8, depth=2)
    net = net.to(device).train()

    x = torch.randn(2, 3, 64, 64, device=device)
    y = torch.zeros(2, N_CLASSES, 64, 64, device=device)
    y[:, 0] = 1.0
    y[:, 0, 20:40, 20:40], y[:, VOID_CLASS, 20:40, 20:40] = 0.0, 1.0

    criterion = BCEDiceLoss().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        loss = criterion(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"

    # A confident all-matrix prediction against a real void must be expensive.
    # If it is cheap the rare class never gets learned, and training still looks
    # healthy because the BCE term keeps dropping.
    crit = BCEDiceLoss(void_weight=5.0)
    logits = torch.full((1, N_CLASSES, 8, 8), -10.0)
    logits[:, 0] = 10.0
    tgt = torch.zeros(1, N_CLASSES, 8, 8)
    tgt[:, 0] = 1.0
    tgt[:, 0, 2:5, 2:5], tgt[:, VOID_CLASS, 2:5, 2:5] = 0.0, 1.0
    missed = crit(logits, tgt).item()

    clean = torch.zeros(1, N_CLASSES, 8, 8)
    clean[:, 0] = 1.0
    assert missed > crit(logits, clean).item() + 0.5, "missing a void was nearly free"

    # The void channel must be weighted above the others, or the imbalance
    # correction is silently absent.
    assert crit.bce.pos_weight[VOID_CLASS].item() == 5.0
    assert crit.bce.pos_weight[0].item() == 1.0
    # Dice must see logits, not probabilities - a second sigmoid is a silent leak.
    assert crit.dice.from_logits

    print(f"ok  loss {losses[0]:.3f} -> {losses[-1]:.3f}  missed-void penalty {missed:.3f}")


if __name__ == "__main__":
    main()
