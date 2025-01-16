import torch
import torch.nn as nn
from torch import Tensor


def init_weights(module: nn.Module):
    """Initialize weights for the module."""
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')


class FirstFeature(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        # self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        # self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_skip: bool = True):
        super(Decoder, self).__init__()
        self.use_skip = use_skip
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_block = ConvBlock(in_channels if use_skip else out_channels, out_channels)
        # self.apply(init_weights)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if self.use_skip:
            x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class FinalOutput(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
        # self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 3,
        features: list[int] = [64, 128, 256, 512, 1024],
        use_skip: bool = True
    ):
        super(UNet, self).__init__()

        self.in_conv1 = FirstFeature(n_channels, features[0])
        self.in_conv2 = ConvBlock(features[0], features[0])

        self.encoders = nn.ModuleList(
            [Encoder(features[i], features[i + 1]) for i in range(len(features) - 1)]
        )

        self.decoders = nn.ModuleList(
            [Decoder(features[i], features[i - 1], use_skip=use_skip) for i in range(len(features) - 1, 0, -1)]
        )

        self.out_conv = FinalOutput(features[0], n_classes)

    def forward(self, x: Tensor) -> Tensor:
        skips = []

        # Encoding
        x = self.in_conv1(x)
        skips.append(self.in_conv2(x))

        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Decoding
        skips = skips[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i + 1])

        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    model = UNet()
    input_tensor = torch.ones(2, 3, 64, 64)
    output = model(input_tensor)
    print("Output Shape:", output.shape)

    model = UNet(use_skip=False)
    output = model(input_tensor)
    print("Output Shape without skip connection:", output.shape)
