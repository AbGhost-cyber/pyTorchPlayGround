import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)


# create some linearly separable data with three classes

class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x > 1.0)[:, 0]] = 2
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


data_set = Data()
# Build Softmax Classifier technically you only need nn.Linear
model = nn.Sequential(nn.Linear(1, 3))
# print(model.state_dict())
# Create criterion function, optimizer, and dataloader
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
trainloader = DataLoader(dataset=data_set, batch_size=5)

LOSS = []


def train_model(epochs):
    for epoch in range(epochs):
        for x, y in trainloader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.detach())
            loss.backward()
            optimizer.step()


# train_model(300)
# Make the prediction
z = model(data_set.x)
_, yhat = z.max(1)
# print("The prediction:", yhat)

correct = (data_set.y == yhat).sum().item()
accuracy = correct / len(data_set)


# print("The accuracy: ", accuracy)

# X = data_set[:][0]
# Y = data_set[:][1]
# plt.plot(X[Y == 0, 0].numpy(), Y[Y == 0].numpy(), 'bo', label='y = 0')
# plt.plot(X[Y == 1, 0].numpy(), 0 * Y[Y == 1].numpy(), 'ro', label='y = 1')
# plt.plot(X[Y == 2, 0].numpy(), 0 * Y[Y == 2].numpy(), 'go', label='y = 2')
# plt.show()

# Exercise:
# Implement a function `custom_softmax()` that takes in a PyTorch tensor `logits` of shape `(batch_size, num_classes)`
# and applies the softmax function to it. The function should return a PyTorch tensor of the same shape, containing
# the probability distribution over the classes for each input data point.
# You should not use the built-in `torch.nn.functional.softmax()` function.

def custom_softmax(logits):
    exp_logits = torch.exp(logits)
    sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
    softmax = exp_logits / sum_exp_logits
    return softmax


logits = torch.tensor(np.random.randn(10, 3))
print(torch.softmax(logits, dim=1))
print(custom_softmax(logits))
if __name__ == '__main__':
    print()
