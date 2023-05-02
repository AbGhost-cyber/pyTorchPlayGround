import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(D_in, H)
        self.output_layer = nn.Linear(H, D_out)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        hidden = self.hidden_layer(x)
        activated1 = self.sigmoid1(hidden)
        output = self.output_layer(activated1)
        activated2 = self.sigmoid2(output)
        return activated2


# make some data
X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0


# in most cases we might just have to pass a layer rather than input size etc

class LayerNet(nn.Module):
    def __init__(self, Layers):
        super(LayerNet, self).__init__()
        self.hidden = nn.ModuleList()
        for i in range(len(Layers) - 1):
            self.hidden.append(nn.Linear(Layers[i], Layers[i + 1]))
            self.hidden.append(nn.ReLU())

    def forward(self, activation):
        length = len(self.hidden)
        for index, linear_transform in zip(range(length), self.hidden):
            if index < length - 1:
                activation = nn.ReLU(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation


layers = [2, 10, 10, 3]
layerNet = LayerNet(Layers=layers)

if __name__ == '__main__':
    print()
