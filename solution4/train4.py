"""Solution 4 training: solution 1, with the augmentation RNG unfrozen.

    python solution4/train4.py --epochs 20 --fold 0
    python solution4/train4.py --demo

A controlled A/B. The loss and the validation metric are imported from
src/train.py rather than copied, the split and augmentation come from
src/data.py, and the model comes from src/model.py. The only difference from
solution 1 is `Micrographs4` in place of `Micrographs`, which changes how the
augmentation RNG is seeded and nothing else.

Defaults match how solution 1's shipped checkpoints were actually produced -
20 epochs, batch 16, AdamW 3e-4, cosine, class weights (1, 1, 5). l4_train.sh
passed --epochs 20 even though src/train.py defaults to 30, so 20 is the
comparable number here.

If this scores no better than solution 1, the augmentation was doing little
even when it was working, and the effort spent designing it was misplaced. If
it scores better, solution 1 has been leaving that much on the table for the
entire competition.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import index_training  # noqa: E402
from data4 import Micrographs4  # noqa: E402
from model import build  # noqa: E402
from train import validate, void_dice_loss  # noqa: E402  - solution 1's own, imported

OUT = Path(__file__).resolve().parent / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20,
                    help="20 matches how solution 1's shipped checkpoints were trained")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--require-cuda", action="store_true",
                    help="Exit rather than fall back to CPU")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        return demo()

    out = Path(args.out or OUT / f"s4_unet_f{args.fold}_s{args.seed}.pt")
    if args.require_cuda and not torch.cuda.is_available():
        sys.exit(f"--require-cuda given but CUDA is unavailable (torch {torch.__version__})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
          f"  torch {torch.__version__}")
    torch.manual_seed(args.seed)

    df = index_training(args.fold)
    train_df, val_df = df[~df.is_val], df[df.is_val]
    if args.limit:
        train_df, val_df = train_df.head(args.limit), val_df.head(max(4, args.limit // 4))
    print(f"fold {args.fold} seed {args.seed}: train {len(train_df)}  val {len(val_df)}")

    common = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.workers > 0)
    # The one changed line: Micrographs4 instead of Micrographs.
    train_loader = DataLoader(Micrographs4(train_df, train=True, seed=args.seed),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(Micrographs4(val_df), batch_size=args.batch_size, **common)

    net, _ = build("unet", args.base, args.depth, chroma=False)
    net = net.to(device)
    print(f"unet base={args.base} depth={args.depth}  "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, len(train_loader)))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    ce_weight = torch.tensor([1.0, 1.0, 5.0], device=device)

    best = -1.0
    out.parent.mkdir(parents=True, exist_ok=True)
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
            # Same fields as solution 1 so predict.py and bench.py can read
            # these checkpoints unchanged.
            torch.save({"model": net.state_dict(), "base": args.base, "depth": args.depth,
                        "arch": "unet", "chroma": False, "val_dice": dice,
                        "fold": args.fold, "seed": args.seed,
                        "pipeline": "solution4-fixed-rng"}, out)
            flag = "  <- saved"
        print(f"epoch {epoch:3d}  loss {running / max(1, len(train_loader)):.4f}  "
              f"val_dice_void {dice:.4f}  {time.time() - started:.0f}s{flag}")

    print(f"\nbest val Dice_void: {best:.4f}  ->  {out}")


def demo():
    """The dataset is the only difference, and it is a real one."""
    from data import Micrographs

    df = index_training(0)
    sub = df[~df.is_val].head(3)

    def distinct(ds):
        seen = {i: set() for i in range(len(ds))}
        for _ in range(5):
            for i in range(len(ds)):
                seen[i].add(hash(ds[i][0].numpy().tobytes()))
        return [len(s) for s in seen.values()]

    old, new = distinct(Micrographs(sub, train=True)), distinct(Micrographs4(sub, train=True))
    assert max(old) == 1 and min(new) >= 4, (old, new)

    # The loss must be solution 1's, not a lookalike - if this import ever
    # breaks, the A/B silently stops being an A/B.
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, :4, :4] = 2
    perfect = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 20
    assert void_dice_loss(perfect, target) < 0.05
    assert void_dice_loss(-perfect, target) > 0.9

    print(f"ok  augmentations per image over 5 draws: solution 1 {old}, solution 4 {new}")


if __name__ == "__main__":
    main()
