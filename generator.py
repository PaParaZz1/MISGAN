import torch
import torch.nn as nn


class G_layer(nn.Module):
    def __init__(self, nfc, min_nfc, in_channels, num_layers):
        super(G_layer, self).__init__()
        layers = []
        N = nfc
        layers.append(self._conv_layer(in_channels, N, 3, 1, 1))
        for i in range(num_layers - 2):
            N = nfc / (1 << (i + 1))
            layers.append(self._conv_layer(int(max(2 * N, min_nfc)), int(max(N, min_nfc)), 3, 1, 1))
        layers.append(self._conv_layer(max(N, min_nfc), in_channels, 3, 1, 1, False, nn.Tanh()))
        self.layers = nn.Sequential(*layers)

    def _conv_layer(self, in_channels, out_channels, kernel, stride, padding, bn=True, active=nn.LeakyReLU(0.2)):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=True))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(active)
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel, stride, padding, mode="avg"):
        if mode == "avg":
            return nn.AvgPool2d(kernel, stride, padding)
        elif mode == "max":
            return nn.MaxPool2d(kernel, stride, padding)
        else:
            assert 0

    def forward(self, x):
        x = self.layers(x)
        return x


class Generator(nn.Module):
    def __init__(self, nfc, min_nfc, N=3, in_channels=3, num_layers=5):
        super(Generator, self).__init__()
        self.N = N
        self.in_channels = in_channels
        self.nfc = nfc
        self.min_nfc = min_nfc
        self.num_layers = num_layers
        self.build()

    def _conv_layer(self, in_channels, out_channels, kernel, stride, padding, bn=True, active=nn.LeakyReLU(0.2)):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=True))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(active)
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel, stride, padding, mode="avg"):
        if mode == "avg":
            return nn.AvgPool2d(kernel, stride, padding)
        elif mode == "max":
            return nn.MaxPool2d(kernel, stride, padding)
        else:
            assert 0

    def build(self):
        layers = []
        for i in range(self.N - 1):
            layers.append(self._conv_layer(self.in_channels, self.in_channels, 3, 2, 1))
        self.head = nn.Sequential(*layers)
        self.G = []
        for i in range(self.N):
            self.G.append(G_layer(self.nfc, self.min_nfc, self.in_channels, self.num_layers))
        self.g=nn.Sequential(*self.G)

    def forward(self, x, size):
        x = nn.UpsamplingBilinear2d(size=size)(x)
        x = self.head(x)
        output = []
        for i in range(self.N):
            x = self.G[i](x)
            output.append(x)
            if i < self.N - 1:
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
        return output


if __name__ == "__main__":
    model = Generator(32, 32, N=3, in_channels=3).cuda()
    torch.save(model.state_dict(),'test.pth')
    x = torch.randn([3, 3, 200, 300]).cuda()
    with torch.no_grad():
        print(model(x, [400, 500])[2].shape)
