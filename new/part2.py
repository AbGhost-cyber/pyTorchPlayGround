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
print(len(data.get_dataLoader(train=False)))
if __name__ == '__main__':
    print()
