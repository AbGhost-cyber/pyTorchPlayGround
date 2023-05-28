import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1)
# param initialization
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.randn(size=(2, 4))


def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(tensor=module.weight, mean=0, std=0.01)
        nn.init.zeros_(tensor=module.bias)


# applies the func recursively to every submodule
net.apply(init_normal)
net(X)
# print(net[0].weight.data[0], net[0].bias.data[0])
if __name__ == '__main__':
    print()


# def apply_init(self, inputs, init=None):
# self.forward(*inputs)
# if init is not None:
#     self.net.apply(init)

# custom layers


class Mylinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_units, units))
        self.bias = nn.Parameter(torch.rand(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


class CenteredLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())

# print(layer(torch.tensor([1.0, 2, 3, 4, 5])))
