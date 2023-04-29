import numpy as np
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from more_linear import LinearRegression

torch.manual_seed(1)

w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)


def forward(x):
    yhat = torch.mm(x, w) + b
    # A * B != B * A, columns of a must equal rows of B
    # print(x.size())
    # print(w.size())
    return yhat


x = torch.tensor([[1.0, 2.0]])
yhat = forward(x)
# print(f"yhat: {yhat}")

model = LinearRegression(2, 1)
# print("The parameters: ", list(model.parameters()))
yhat = model(x)

# Practice
X = torch.tensor([[11.0, 12.0, 13, 14], [11, 12, 13, 14]])
model = LinearRegression(4, 1)
yhat = model(X)


# print(yhat)

# Write a program that performs linear regression on a toy dataset containing 10 samples with 1 input feature and
# 1 output feature. Your implementation should use PyTorch tensors and the mean squared error (MSE) loss function.
class Data(Dataset):
    def __init__(self, train=True):
        # Generate toy dataset
        self.x = torch.randn(10, 1)
        self.f = -3 * self.x + 1
        self.y = self.f + 0.2 * torch.randn(self.x.size())
        self.len = self.x.shape[0]

        # outliers
        if train:
            self.y[0] = 0
            self.y[50:55] = 0
        else:
            pass

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


class ToyLR(nn.Module):

    def __init__(self, in_size, out_size):
        super(ToyLR, self).__init__()
        self.linear = nn.Linear(in_features=in_size, out_features=out_size)

    def forward(self, x):
        return self.linear(x)


train_data = Data()
val_data = Data(train=False)
toy_model = ToyLR(in_size=1, out_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(toy_model.parameters(), lr=0.1)
trainLoader = DataLoader(train_data, batch_size=1)
LOSS = []


def train_toy(epochs):
    for epoch in range(epochs):
        for x, y in trainLoader:
            yhat = toy_model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# train_toy(10)
# print(LOSS)

# we could try with multiple learning rates
learning_rates = [0.0001, 0.001, 0.01, 0.1]
train_error = torch.zeros(len(learning_rates))
validation_error = torch.zeros(len(learning_rates))
MODELS = []


def train_model_with_lr(epochs):
    for i, lr in enumerate(learning_rates):
        toy_model = ToyLR(1, 1)
        optimizer = optim.SGD(toy_model.parameters(), lr=lr)
        for epoch in range(epochs):
            for x, y in trainLoader:
                yhat = toy_model(x)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # train data
        Yhat = toy_model(train_data.x)
        train_loss = criterion(Yhat, train_data.y)
        train_error[i] = train_loss.item()
        # validation data
        Yhat = toy_model(val_data.x)
        val_loss = criterion(Yhat, val_data.y)
        validation_error[i] = val_loss.item()
        MODELS.append(model)


train_model_with_lr(10)

# Plot the training loss and validation loss
plt.semilogx(np.array(learning_rates), train_error.numpy(), label='training loss/total Loss')
plt.semilogx(np.array(learning_rates), validation_error.numpy(), label='validation cost/total Loss')
plt.ylabel('Cost / Total Loss')
plt.xlabel('learning rate')
plt.legend()
plt.show()

# Plot the predictions

i = 0
for model, learning_rate in zip(MODELS, learning_rates):
    yhat = toy_model(val_data.x)
    plt.plot(val_data.x.numpy(), yhat.detach().numpy(), label='lr:' + str(learning_rate))
    print('i', yhat.detach().numpy()[0:3])
plt.plot(val_data.x.numpy(), val_data.f.numpy(), 'or', label='validation data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

if __name__ == '__main__':
    print()
