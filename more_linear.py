import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class Data(Dataset):

    # Constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.f = 1 * self.x - 1
        self.y = self.f + 0.1 * torch.randn(self.x.size())
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len


class LinearRegression(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=in_size, out_features=out_size)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression(in_size=1, out_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
trainLoader = DataLoader(dataset=Data(), batch_size=1)
# since pyTorch assigns random values for the params, we can mutate
model.state_dict()['linear.weight'][0] = -15
model.state_dict()['linear.bias'][0] = -10
print(model.state_dict())


def train_model_BGD(epochs):
    for epoch in range(epochs):
        for x, y in trainLoader:
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    print()
