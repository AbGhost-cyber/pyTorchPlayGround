import matplotlib.pyplot as plt
import torch
import torch.nn as nn

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)


# plt.plot(x.detach().numpy(), y.detach().numpy())
# plt.ylabel('relu(x)')
# plt.xlabel('x')
# plt.show()
# we can also plot the derivative
# y.backward(torch.ones_like(x), retain_graph=True)
# plt.plot(x.detach(), x.grad)
# plt.xlabel('x')
# plt.ylabel('grad of relu')
# plt.show()
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# print(relu(torch.tensor([1, 2, -1])))

class MLP(nn.Module):
    def __init__(self, num_hidden, num_outputs):
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(out_features=num_hidden),
                                 nn.ReLU(),
                                 nn.LazyLinear(out_features=num_outputs))

    def forward(self, X):
        return self.net(X)


mlp = MLP(num_hidden=3, num_outputs=1)
yhat = mlp(torch.tensor([[1, -1.0, -2.4], [3, 4, 5]]))
# print(yhat)


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # check if we want to apply dropout to the entire neurons
    if dropout == 1:
        return torch.ones_like(X)
    mask = (torch.randn(X.shape) > dropout).float()
    # we scale up the outputs by a factor of 1 / (1 - dropout) to compensate for the neurons that are dropped out.
    return mask * X / (1.0 - dropout)


X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print((torch.randn(X.shape) > 0.6).float() * X / (1.0 - 0.6))
if __name__ == '__main__':
    print()
