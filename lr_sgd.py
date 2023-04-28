import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Set random seed
torch.manual_seed(1)
# Set up the actual data and simulated data
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())

w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)


# predict
def forward(x):
    yhat = w * x + b
    return yhat


# loss function MSE
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)


lr = 0.1
# batch gradient descent losses
LOSS_BGD = []
# stochastic gradient descent losses
LOSS_SGD = []


def train_model(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        loss = criterion(Yhat, Y)
        LOSS_BGD.append(loss.detach())
        loss.backward()
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()


def train_model_SGD(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        LOSS_SGD.append(criterion(Yhat, Y).tolist())
        for x, y in zip(X, Y):
            yhat = forward(x)
            loss = criterion(yhat, y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()


# train_model(10)
# train_model_SGD(10)

# plt.plot(LOSS_BGD, label="Batch Gradient Descent")
# plt.plot(LOSS_SGD, label="Stochastic Gradient Descent")
# plt.xlabel('epoch')
# plt.ylabel('Cost/ total loss')
# plt.legend()
# plt.show()

# we can use a dataLoader with SGD instead

class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * self.x - 1
        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


# create the dataset
data_set = Data()
# create data loader
trainLoader = DataLoader(dataset=data_set, batch_size=1)


def train_model_DataLoader(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        LOSS_SGD.append(criterion(Yhat, Y).tolist())
        for x, y in trainLoader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()


train_model_DataLoader(10)
if __name__ == '__main__':
    print()
