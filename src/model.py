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
    def __init__(self, n_classes=N_CLASSES, base=32, depth=4):
        super().__init__()
        chans = [base * 2**i for i in range(depth)]

        self.downs = nn.ModuleList()
        cin = N_CHANNELS
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
    "unet_r34":     ("Unet", "resnet34"),        # pretrained encoder, same decoder shape
    "unetpp_r34":   ("UnetPlusPlus", "resnet34"),  # nested skips, aimed at fine detail
    "unet_effb0":   ("Unet", "efficientnet-b0"),   # different inductive bias, lighter
    "fpn_r34":      ("FPN", "resnet34"),           # multi-scale head
}

# ImageNet encoders were trained on normalised input. Feeding them raw [0,1]
# throws away the pretraining that is the entire reason to use them.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _Normalised(nn.Module):
    """Applies ImageNet statistics before the wrapped network sees the input.

    Kept inside the module on purpose: normalisation that lives in the
    training script is normalisation inference forgets, and a pretrained
    encoder fed raw [0,1] silently throws away the weights it was chosen for.
    As buffers these travel in the state dict, so a checkpoint is self-contained.
    """

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build(arch="unet", base=32, depth=4):
    """The scratch U-Net, or a named segmentation_models_pytorch variant."""
    if arch == "unet":
        return UNet(base=base, depth=depth), False

    import segmentation_models_pytorch as smp

    cls, encoder = SMP_ARCHS[arch]
    net = getattr(smp, cls)(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=N_CHANNELS,
        classes=N_CLASSES,
    )
    return _Normalised(net, IMAGENET_MEAN, IMAGENET_STD), True


def demo():
    net = UNet()
    y = net(torch.zeros(2, N_CHANNELS, 256, 256))
    assert y.shape == (2, N_CLASSES, 256, 256), y.shape
    # Fully convolutional: any size that survives 4 halvings must also work,
    # which is what lets inference run on whole images without tiling.
    assert net(torch.zeros(1, N_CHANNELS, 128, 192)).shape == (1, N_CLASSES, 128, 192)
    print(f"ok  params={sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")


if __name__ == "__main__":
    demo()
