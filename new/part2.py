import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

torch.manual_seed(1)


class SyntheticRegressionData:
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        n = num_train + num_val
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise
        self.len = self.X.shape[0]

    def get_dataloader(self, train):
        if train:
            indices = list(range(0, self.num_train))
            random.shuffle(indices)
        else:
            indices = list(range(self.num_train, self.num_train + self.num_val))
        for i in range(0, len(indices), self.batch_size):
            batch_indices = torch.tensor(indices[i: i + self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]

    # above method is not good enough, so we use pyTorch iterator

    def get_tensorLoader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=train)

    def get_dataLoader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorLoader((self.X, self.y), train, i)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=2)


# print('features:', data.X[0], '\nlabel:', data.y[0])
# X, y = next(iter(data.get_dataloader(train=True)))
# print('X shape:', X.shape, '\ny shape:', y.shape)
# print(len(data.get_dataLoader(train=False)))

class LinearRegressionScratch(nn.Module):
    def __init__(self, num_inputs, lr, sigma=0.01):
        super(LinearRegressionScratch, self).__init__()
        self.lr = lr
        self.w = torch.normal(mean=0.0, std=sigma, size=(num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, yhat, y):
        l = (yhat - y) ** 2 / 2
        return l.mean()

    def optim(self):
        return SGD(params=[self.w, self.b], lr=self.lr)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        # self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        # self.plot('loss', l, train=False)
        return l


class SGD:

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


model = LinearRegressionScratch(num_inputs=2, lr=0.03)
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)


def train_model():
    model.train()
    for batch in data.train_dataloader():
        loss = model.training_step(batch)
        model.optim().zero_grad()
        with torch.no_grad():
            loss.backward()



if __name__ == '__main__':
    print()
