import torch
import torch.nn as nn
from matplotlib import pyplot as plt

torch.manual_seed(2)

z = torch.arange(-100, 100, 0.1).view(-1, 1)
sig = nn.Sigmoid()
yhat = sig(z)
# we can plot
# plt.plot(z.numpy(), yhat.numpy())
# plt.xlabel('z')
# plt.ylabel('yhat')

model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
X = torch.tensor([[1.0], [100]])
yhat = model(X)
print(yhat)
if __name__ == '__main__':
    print()
