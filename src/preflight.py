"""Check the environment before committing an hour to a run.

    python src/preflight.py           # training + evaluation
    python src/preflight.py --eval    # evaluation only (no GPU needed)

Exits non-zero if anything required is missing, so it can gate a script:

    python src/preflight.py || exit 1

`segmentation_models_pytorch` is required wherever the pretrained
architectures are *loaded*, not just where they are trained - reconstructing
an arch_* checkpoint goes through smp on the evaluating machine too.
"""

import argparse
import importlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (import name, pip name, why)
REQUIRED = [
    ("numpy", "numpy", "arrays"),
    ("scipy", "scipy", "severity geometry (KD-trees, convex hulls)"),
    ("pandas", "pandas", "metadata.csv"),
    ("PIL", "pillow", "image and mask IO"),
    ("skimage", "scikit-image", "connected components, despeckling"),
    ("torch", "torch", "the model"),
]
OPTIONAL = [
    ("segmentation_models_pytorch", "segmentation_models_pytorch",
     "pretrained architectures - needed to TRAIN and to LOAD arch_* checkpoints"),
    ("torchvision", "torchvision", "pulled in by smp via timm; must match torch's build"),
]


def check_imports(specs, required):
    ok = True
    for mod, pip_name, why in specs:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [ok]      {mod:32} {ver:14} {why}")
        except ImportError:
            tag = "MISSING" if required else "absent"
            print(f"  [{tag:5}]  {mod:32} {'':14} {why}")
            print(f"            -> pip install {pip_name}")
            ok = ok and not required
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="Evaluation only; do not require a GPU")
    args = ap.parse_args()

    print(f"python    {sys.version.split()[0]}  ({sys.platform})")
    print(f"repo      {REPO}\n")

    print("required packages")
    ok = check_imports(REQUIRED, required=True)

    print("\noptional packages")
    check_imports(OPTIONAL, required=False)

    # Installing smp pulls torchvision, and on Windows the default PyPI torch
    # wheel is CPU-only - so a plain `pip install segmentation_models_pytorch`
    # can silently replace a CUDA build with a CPU one, or leave torchvision
    # built against a torch that is no longer installed. Both fail confusingly
    # much later, so check the pair here.
    try:
        import torch
        import torchvision
        tv_base = torchvision.__version__.split("+")[0]
        t_tag = torch.__version__.split("+")[-1] if "+" in torch.__version__ else "cpu"
        tv_tag = torchvision.__version__.split("+")[-1] if "+" in torchvision.__version__ else "cpu"
        if t_tag != tv_tag:
            print(f"\n  [MISMATCH] torch is '{t_tag}' but torchvision is '{tv_tag}'")
            print(f"            -> pip install torch torchvision --index-url "
                  f"https://download.pytorch.org/whl/{t_tag}")
            ok = False
        else:
            print(f"\n  [ok]      torch/torchvision builds agree ({t_tag}, torchvision {tv_base})")
    except ImportError:
        pass

    print("\ngpu")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  [ok]      {torch.cuda.get_device_name(0)}  "
                  f"{total / 1024**3:.1f} GB ({free / 1024**3:.1f} free)  torch {torch.__version__}")
        elif args.eval:
            print(f"  [ok]      no CUDA - fine for evaluation, which is CPU-bound anyway")
        else:
            print(f"  [MISSING] torch {torch.__version__} sees no CUDA device; training would be very slow")
            ok = False
    except ImportError:
        print("  [MISSING] torch not installed")
        ok = False

    print("\ndata")
    sets = ["Data set I", "Data set II", "Data set III", "Test data set"]
    for name in sets:
        d = REPO / "data" / "Data sets" / name
        n = len(list(d.rglob("*.png"))) + len(list(d.rglob("*.tif"))) + len(list(d.rglob("*.jpg")))
        flag = "ok" if n else "MISSING"
        print(f"  [{flag:5}]  {name:32} {n} files")
        ok = ok and bool(n)

    if not (REPO / "evaluation.py").exists():
        print("  [MISSING] evaluation.py - the scoring functions are imported from it")
        ok = False
    else:
        print("  [ok]      evaluation.py")

    print("\ndisk")
    free_gb = shutil.disk_usage(REPO).free / 1024**3
    flag = "ok" if free_gb > 3 else "LOW"
    print(f"  [{flag:5}]  {free_gb:.1f} GB free  (checkpoints are ~30-100 MB each, 10 of them)")

    print("\n" + ("PREFLIGHT OK" if ok else "PREFLIGHT FAILED - fix the MISSING lines above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
