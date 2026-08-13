"""A small U-Net. RGB in, 3 class logits out.

Deliberately plain: no pretrained backbone, no attention, no deep supervision.
The dataset is 4000 small images, so a ~2M parameter U-Net trains in minutes
on one GPU and there is nothing here for a bigger model to learn that this one
cannot.
"""

import torch
import torch.nn as nn

from data import N_CHANNELS, N_CLASSES


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES, base=32, depth=4, in_ch=N_CHANNELS):
        super().__init__()
        chans = [base * 2**i for i in range(depth)]

        self.downs = nn.ModuleList()
        cin = in_ch
        for c in chans:
            self.downs.append(_block(cin, c))
            cin = c

        self.bottom = _block(cin, cin * 2)
        cin *= 2

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for c in reversed(chans):
            self.ups.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.up_convs.append(_block(c * 2, c))
            cin = c

        self.head = nn.Conv2d(cin, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottom(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([x, skip], dim=1))

        return self.head(x)


#: Architectures for the comparison. Each is (smp class, encoder), except
#: "unet" which is the scratch model above. ImageNet-pretrained encoders are
#: the point of the exercise: if none of them clears the Dice gate either,
#: the ceiling is in the labels rather than the model.
SMP_ARCHS = {
    "unet_r34":     ("Unet", "resnet34"),          # pretrained encoder, same decoder shape
    "unetpp_r34":   ("UnetPlusPlus", "resnet34"),  # nested skips, aimed at fine detail
    "unet_effb0":   ("Unet", "efficientnet-b0"),   # different inductive bias, lighter
    "fpn_r34":      ("FPN", "resnet34"),           # multi-scale head
    # Positive control, expected to lose. DeepLabV3+ predicts at output stride
    # 4 and bilinearly upsamples 4x, so a 15px median void is ~4px when its
    # boundary is decided and interpolated afterwards - and 1px of boundary
    # drift on a void that size is a ~13% Dice hit. Its ASPP is built for
    # large-receptive-field scene parsing, the opposite of small high-contrast
    # blobs. If everything ties at 0.744 AND this one drops, the comparison
    # demonstrably resolves architecture differences, so the tie is a real
    # ceiling rather than a blunt instrument.
    "deeplabv3p_r34": ("DeepLabV3Plus", "resnet34"),
}

# ImageNet encoders were trained on normalised input. Feeding them raw [0,1]
# throws away the pretraining that is the entire reason to use them.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _InputAdapter(nn.Module):
    """Turns raw RGB in [0,1] into whatever the wrapped network expects.

    Two transforms, both kept inside the module on purpose. Normalisation that
    lives in the training script is normalisation inference forgets, and a
    pretrained encoder fed raw [0,1] silently discards the weights it was
    chosen for. As buffers these travel in the state dict, so a checkpoint
    stays self-contained and predict/bench/report never learn about any of it.

    Chromaticity (r = R/(R+G+B)) is appended rather than substituted. Voids are
    darker *and* bluer, so replacing RGB would throw away the intensity half of
    the signal; the network gets both. Only two ratios are added because the
    third is 1 - r - g, and a linear function of existing channels is exactly
    what the first convolution can already form for itself.

    The division is the point. A 3x3 conv over RGB can produce any linear
    combination - B-R, greyscale, a LAB-style axis - so handing it engineered
    linear features gains nothing. Dividing by the per-pixel sum is non-linear,
    cannot be expressed as a convolution, and removes illumination while
    keeping hue. That directly targets the measured gap: the Test set is more
    saturated than training, 37 vs 25 mean channel spread.
    """

    def __init__(self, net, chroma=False, mean=None, std=None):
        super().__init__()
        self.net = net
        self.chroma = chroma
        self.normalise = mean is not None
        if self.normalise:
            self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        # Chromaticity comes off the raw signal, before any normalisation
        # shifts what "sum of channels" means.
        extra = x[:, :2] / x.sum(1, keepdim=True).clamp(min=1e-4) if self.chroma else None
        if self.normalise:
            x = (x - self.mean) / self.std
        return self.net(x if extra is None else torch.cat([x, extra], dim=1))


def build(arch="unet", base=32, depth=4, chroma=False):
    """The scratch U-Net, or a named segmentation_models_pytorch variant."""
    in_ch = N_CHANNELS + 2 if chroma else N_CHANNELS

    if arch == "unet":
        net = UNet(base=base, depth=depth, in_ch=in_ch)
        return (_InputAdapter(net, chroma=True) if chroma else net), False

    import segmentation_models_pytorch as smp

    cls, encoder = SMP_ARCHS[arch]
    net = getattr(smp, cls)(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_ch,
        classes=N_CLASSES,
    )
    return _InputAdapter(net, chroma, IMAGENET_MEAN, IMAGENET_STD), True


def demo():
    net = UNet()
    y = net(torch.zeros(2, N_CHANNELS, 256, 256))
    assert y.shape == (2, N_CLASSES, 256, 256), y.shape
    # Fully convolutional: any size that survives 4 halvings must also work,
    # which is what lets inference run on whole images without tiling.
    assert net(torch.zeros(1, N_CHANNELS, 128, 192)).shape == (1, N_CLASSES, 128, 192)

    # The chromaticity channels must be invariant to illumination - that is the
    # entire reason they are not just another linear combination of RGB. Halve
    # the brightness and they must not move, while RGB obviously does.
    adapter = _InputAdapter(nn.Identity(), chroma=True)
    x = torch.rand(2, N_CHANNELS, 8, 8) * 0.8 + 0.1
    bright, dim = adapter(x), adapter(x * 0.5)
    assert torch.allclose(bright[:, 3:], dim[:, 3:], atol=1e-5), "chromaticity is not illumination-invariant"
    assert not torch.allclose(bright[:, :3], dim[:, :3]), "RGB should still carry intensity"
    assert bright.shape[1] == N_CHANNELS + 2

    # Black pixels must not divide by zero.
    assert torch.isfinite(adapter(torch.zeros(1, N_CHANNELS, 4, 4))).all()

    print(f"ok  params={sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")


if __name__ == "__main__":
    demo()
