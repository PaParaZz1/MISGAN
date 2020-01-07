import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, N=3, in_channels=3):
        super(Generator, self).__init__()
        self.N = N
        self.in_channels = in_channels
        self.build()

    def _conv_layer(self, in_channels, out_channels, kernel, stride, padding, bn=True, active=nn.ReLU):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=True))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(active())
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel, stride, padding, mode="avg"):
        if mode == "avg":
            return nn.AvgPool2d(kernel, stride, padding)
        elif mode == "max":
            return nn.MaxPool2d(kernel, stride, padding)
        else:
            assert 0

    def G_layer(self, channels):
        c1 = self._conv_layer(channels, channels, 3, 1, 1)
        c2 = self._conv_layer(channels, channels, 3, 1, 1)
        c3 = self._conv_layer(channels, channels, 3, 1, 1)
        c4 = self._conv_layer(channels, channels, 3, 1, 1)
        return nn.Sequential(c1, c2, c3, c4)

    def build(self):
        self.head = self._conv_layer(3, 8, 3, 1, 1)
        self.G = []
        self.P = []
        self.F = []
        self.G.append(self.G_layer(8))
        self.P.append(self._conv_layer(8, 16, 3, 2, 1))
        self.G.append(self.G_layer(16))
        self.P.append(self._conv_layer(16, 32, 3, 2, 1))
        self.G.append(self.G_layer(32))
        self.F.append(self._conv_layer(32, 16, 3, 1, 1))
        self.F.append(self._conv_layer(32, 8, 3, 1, 1))
        self.F.append(self._conv_layer(16, 3, 3, 1, 1))
        self.g = nn.Sequential(*self.G)
        self.p = nn.Sequential(*self.P)
        self.f = nn.Sequential(*self.F)

    def forward(self, x, size):
        x = nn.UpsamplingBilinear2d(size=size)(x)
        x = self.head(x)
        y = []
        for i in range(self.N):
            y.append(self.G[i](x))
            if i < self.N - 1:
                x = self.P[i](x)
        x = y[self.N - 1]
        for i in range(self.N):
            x = self.F[i](x)
            if i < self.N - 1:
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = torch.cat([x, y[self.N - i - 2]], 1)
        return x


if __name__ == "__main__":
    model = Generator(N=3, in_channels=3)
    x = torch.randn([3, 3, 200, 300])
    with torch.no_grad():
        print(model(x, [400, 500]).shape)
