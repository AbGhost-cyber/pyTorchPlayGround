import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, TensorDataset, DataLoader


# higher dimensional linear regression practice


class Data(Dataset):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.num_train = num_train
        self.num_val = num_val
        self.num_inputs = num_inputs
        self.batch_size = batch_size
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_tensorLoader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=train)

    def get_dataLoader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorLoader(tensors=[self.X, self.y], train=train, indices=i)

    def train_dataLoader(self):
        return self.get_dataLoader(train=True)

    def val_dataloader(self):
        return self.get_dataLoader(train=False)


class LinearRegressionScratch(nn.Module):
    def __init__(self, num_inputs, lr, sigma=0.01):
        super(LinearRegressionScratch, self).__init__()
        self.lr = lr
        self.w = torch.normal(mean=0.0, std=sigma, size=(num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def l2_penalty(self):
        return (self.w ** 2).sum() / 2

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
        return l
