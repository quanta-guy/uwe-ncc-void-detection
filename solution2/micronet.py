"""MicroNet-pretrained encoders from NASA, loaded into our own smp models.

    python solution2/micronet.py           # self-check: download and load resnet50

https://github.com/nasa/pretrained-microscopy-models (MIT). The encoders are
pretrained on MicroscopyNet - a large corpus of microscopy images - rather than
on ImageNet.

That distinction is the whole reason this is worth running. Every ImageNet
encoder tried on this data lost to the scratch U-Net (0.8446 against 0.8727 on
fold 0), and the explanation offered was domain gap: natural-image priors do
not transfer to micrographs. MicroNet is the control for that claim. Same
architecture, same augmentation, same split, same schedule - only the
pretraining corpus differs. If it wins, the domain-gap explanation holds and we
have a better model. If it loses too, pretraining was never the lever and the
ceiling is in the labels, which is the more valuable finding.

Weights are fetched straight from NASA's public S3 and loaded into an smp model
we construct ourselves. Their package is installed with --no-deps and its
helpers are deliberately not used: it pins segmentation-models-pytorch==0.2.1,
and letting pip act on that would downgrade the 0.5.0 our other models were
built against and take opencv-python-headless with it.
"""

import sys
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data import N_CLASSES  # noqa: E402

S3 = "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"

#: MicroNet's own channel statistics, from pmm.util.get_special_preprocessing_fn.
#: NOT ImageNet's. These encoders were normalised this way in pretraining, and
#: feeding them ImageNet statistics wastes part of what the pretraining bought -
#: which is the exact mistake this experiment exists to test for.
MICRONET_MEAN = (0.4723, 0.4599, 0.4468)
MICRONET_STD = (0.1684, 0.1575, 0.1675)


def weights_url(encoder="resnet50", weights="micronet"):
    """Public S3 URL for a pretrained encoder.

    Only resnet50/micronet has a v1.1; everything else is v1.0. resnext101 is
    published under a shortened name, which upstream also special-cases.
    """
    if encoder == "resnext101_32x8d":
        return S3 + "resnext101_pretrained_microscopynet_v1.0.pth.tar"

    corpus = {"micronet": "microscopynet",
              "image-micronet": "imagenet-microscopynet"}[weights]
    version = "1.1" if (encoder == "resnet50" and weights == "micronet") else "1.0"
    return S3 + f"{encoder}_pretrained_{corpus}_v{version}.pth.tar"


def load_encoder(net, encoder="resnet50", weights="micronet"):
    """Load MicroNet weights into an smp model's encoder, in place.

    The checkpoint carries the classifier head (fc.weight, fc.bias) that the
    encoder has no slot for. Those two are dropped; everything else must match
    exactly, and this asserts that it does rather than trusting strict=False to
    have found anything - a silent partial load would look like training from
    scratch while claiming to be pretrained, and the whole experiment turns on
    the weights actually being there.
    """
    sd = torch.hub.load_state_dict_from_url(weights_url(encoder, weights),
                                            map_location="cpu", progress=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    # nn.DataParallel checkpoints carry a 'module.' prefix the encoder lacks.
    if next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    wanted = set(net.encoder.state_dict())
    sd = {k: v for k, v in sd.items() if k in wanted}
    missing = wanted - set(sd)
    assert not missing, f"{len(missing)} encoder tensors absent from the checkpoint: {sorted(missing)[:5]}"

    net.encoder.load_state_dict(sd)
    return len(sd)


def build_micronet(arch="Unet", encoder="resnet50", weights="micronet", classes=N_CLASSES):
    """An smp model with a MicroNet-pretrained encoder.

    Built here rather than through src/model.build() on purpose: build() wraps
    smp architectures in _InputAdapter, which normalises internally, and the
    albumentations Compose already normalises. Constructing it directly keeps
    exactly one normalisation in the system.
    """
    net = getattr(smp, arch)(encoder_name=encoder, encoder_weights=None,
                             in_channels=3, classes=classes)
    n = load_encoder(net, encoder, weights)
    print(f"  {arch}/{encoder} <- {weights}: {n} encoder tensors loaded")
    return net


def demo():
    """The weights must download, load completely, and actually change the model."""
    net = getattr(smp, "Unet")(encoder_name="resnet50", encoder_weights=None,
                               in_channels=3, classes=N_CLASSES)
    before = net.encoder.conv1.weight.detach().clone()

    n = load_encoder(net, "resnet50", "micronet")
    after = net.encoder.conv1.weight.detach()

    assert n == len(net.encoder.state_dict()), f"loaded {n} of {len(net.encoder.state_dict())}"
    # A load that silently no-ops leaves random init in place, and the run would
    # look like a pretrained model while being nothing of the kind.
    assert not torch.allclose(before, after), "weights did not change - encoder was not loaded"

    y = net(torch.zeros(1, 3, 256, 256))
    assert y.shape == (1, N_CLASSES, 256, 256), y.shape

    assert weights_url("resnet50", "micronet").endswith("microscopynet_v1.1.pth.tar")
    assert weights_url("resnet34", "micronet").endswith("microscopynet_v1.0.pth.tar")
    assert "imagenet-microscopynet" in weights_url("resnet50", "image-micronet")

    print(f"ok  {n} tensors, conv1 delta {(after - before).abs().mean():.5f}, out {tuple(y.shape)}")


if __name__ == "__main__":
    demo()
