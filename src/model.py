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
