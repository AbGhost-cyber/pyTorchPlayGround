import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualNet(nn.Module):
    def __init__(self, num_channels, use1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)

        if (use1x1conv):
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


# blk = ResidualNet(num_channels=6,use1x1conv=True, strides=2)
# X = torch.randn(4, 3, 6, 6)
# pred = blk(X)
def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        layer = []
        for num in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        print(f"layer {self.net}")
        for block in self.net:
            Y = block(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X


if __name__ == '__main__':
    denseBlock = DenseBlock(num_convs=2, num_channels=10)
    X = torch.rand((4, 3, 2, 2))
    denseBlock(X)
