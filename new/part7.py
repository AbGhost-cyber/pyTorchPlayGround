import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1)
# param initialization
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))


# X = torch.randn(size=(2, 4))


def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(tensor=module.weight, mean=0, std=0.01)
        nn.init.zeros_(tensor=module.bias)


# applies the func recursively to every submodule
# net.apply(init_normal)
# net(X)
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

def corr2d(X, kernel):
    # compute size of feature map
    h_out = X.shape[0] - kernel.shape[0] + 1
    w_out = X.shape[1] - kernel.shape[1] + 1
    # init feature map with size
    feature_map = torch.zeros(size=(h_out, w_out))
    for x in range(h_out):
        for y in range(w_out):
            patch = X[x:x + kernel.shape[0], y:y + kernel.shape[1]]
            feature_map[x, y] = torch.sum(patch * kernel)
    return feature_map


# X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
#

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# edge detection
X = torch.ones((6, 8))
X[:, 2:6] = 0
# our kernel
K = torch.tensor([[1.0, -1.0]])
# perform the cross-correlation
Y = corr2d(X, K)
plt.imshow(Y.numpy(), cmap='gray')
plt.show()
# we detect 1 for the edge from white to black and -1 for the edge from black to white. All other outputs take value 0.
# if we apply the kernel to the transposed form of the image it will vanish because the Kernel K only
# detect vertical edges
