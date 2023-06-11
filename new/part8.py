import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, lr=0.1, num_classes=10, *args, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes))
        # use xavier uniform initialization for weights
        # for m in self.modules():
        #     if isinstance(m, nn.LazyConv2d) or isinstance(m, nn.LazyLinear):
        #         nn.init.xavier_uniform_(m.weight)
        if type(nn.Module) == nn.Linear or type(nn.Module) == nn.Conv2d:
            nn.init.xavier_uniform_(nn.Module.weight)

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


# X_shape = torch.tensor([[1, 1, 2], [2, 3, 5]])
# x = torch.randn(*X_shape.shape)
# print("shape: ", X_shape.shape)
# print(x)
model = LeNet()
model.layer_summary((1, 1, 28, 28))
if __name__ == '__main__':
    print()
