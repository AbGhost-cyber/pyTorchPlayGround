import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, TensorDataset, DataLoader


# higher dimensional linear regression practice
def l2_penalty(w):
    return (w ** 2).sum() / 2


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


class LinearRegression(nn.Module):
    def __init__(self, lr):
        super(LinearRegression, self).__init__()
        self.net = nn.LazyLinear(out_features=1)
        self.net.weight.data.normal_(0, 0.01)
        self.lr = lr
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return SGD(params=self.parameters(), lr=self.lr)

    def optim(self):
        return SGD(params=self.parameters(), lr=self.lr)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        # self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l


class WeightDecayScratch(LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        # The value of `lambd` can be set by the user of the model to adjust the strength of the regularization.
        # A higher value of `lambd` will result in stronger regularization, which can help prevent overfitting
        # but may also lead to underfitting if set too high
        self.lambd = lambd

    def loss(self, yhat, y):
        return super().loss(yhat, y) + self.lambd * l2_penalty(self.w)


class WeightDecay(LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr=lr)
        self.wd = wd

    def configure_optimizers(self):
        return SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)

    def get_w_b(self):
        return [self.net.weight, self.net.bias]


# The following code fits our model on the training set with 20 examples and
# evaluates it on the validation set with 100 examples.
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
lambd = 40
lr = 0.01
model = WeightDecay(wd=3, lr=lr)
max_epochs = 10

TRAIN_LOSS = []
VAL_LOSS = []
for epoch in range(max_epochs):
    model.train()
    for batch in data.train_dataLoader():
        loss = model.training_step(batch)
        TRAIN_LOSS.append(loss.detach())
        model.optim().zero_grad()
        with torch.no_grad():
            loss.backward()
            model.optim().step()
    if data.val_dataloader() is None:
        pass
    model.eval()
    for batch in data.val_dataloader():
        with torch.no_grad():
            val_loss = model.validation_step(batch)
            VAL_LOSS.append(val_loss)
print(model.get_w_b()[0])
print('L2 norm of w:', float(l2_penalty(w=model.get_w_b()[0])))

plt.plot(np.log(TRAIN_LOSS))
plt.plot(np.log(VAL_LOSS), linestyle='dotted')
plt.show()

if __name__ == '__main__':
    print()
