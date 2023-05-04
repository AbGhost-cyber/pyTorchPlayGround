from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


class LayerNet(nn.Module):
    def __init__(self, Layers, p=0.0):
        super(LayerNet, self).__init__()
        self.hidden = nn.ModuleList()
        self.drop = nn.Dropout(p=p)
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, activation):
        length = len(self.hidden)
        for index, linear_transform in zip(range(length), self.hidden):
            if index < length - 1:
                activation = F.relu(self.drop(linear_transform(activation)))
            else:
                activation = linear_transform(activation)
        return activation


class Co2EmissionData(Dataset):
    def __init__(self, Train=True):
        pd_data = pd.read_csv("FuelConsumptionCo2.csv")
        pd_data = pd_data.dropna()
        X = pd_data.drop(columns=['MAKE', 'CO2EMISSIONS', 'VEHICLECLASS', 'FUELTYPE', 'MODEL', 'TRANSMISSION']).values
        Y = pd_data['CO2EMISSIONS'].values

        # pre-processing
        X = StandardScaler().fit_transform(X.astype(float))

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        self.x = torch.tensor(X_train if Train else X_test, dtype=torch.float32)
        self.y = torch.tensor(y_train if Train else y_test, dtype=torch.float32)

        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


nn_Layers = [7, 10, 10, 1]
train_dataset = Co2EmissionData()
validation_dataset = Co2EmissionData(Train=False)
co2_emission_model = LayerNet(Layers=nn_Layers)
co2_emission_model_drop = LayerNet(Layers=nn_Layers, p=0.3)
co2_emission_model_drop.train()
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.Adam(params=co2_emission_model.parameters(), lr=learning_rate)
optimizer_drop = optim.Adam(params=co2_emission_model_drop.parameters(), lr=learning_rate)


def train_co2_model(epochs):
    LOSS = {'training data no dropout': [], 'validation data no dropout': [], 'training data dropout': [],
            'validation data dropout': []}
    for epoch in range(epochs):
        yhat = co2_emission_model(train_dataset.x)
        yhat_drop = co2_emission_model_drop(train_dataset.x)
        loss = criterion(yhat, train_dataset.y.unsqueeze(1))
        loss_drop = criterion(yhat_drop, train_dataset.y.unsqueeze(1))

        # store the loss for both models
        LOSS['training data no dropout'].append(loss.item())
        LOSS['validation data no dropout'].append(
            criterion(co2_emission_model(validation_dataset.x), validation_dataset.y.unsqueeze(1)).item())
        LOSS['training data dropout'].append(loss_drop.item())
        co2_emission_model_drop.eval()
        LOSS['validation data dropout'].append(
            criterion(co2_emission_model_drop(validation_dataset.x), validation_dataset.y.unsqueeze(1)).item())
        co2_emission_model_drop.train()

        optimizer.zero_grad()
        optimizer_drop.zero_grad()
        loss.backward()
        loss_drop.backward()
        optimizer.step()
        optimizer_drop.step()
    return LOSS


training_results = train_co2_model(epochs=70)
yhat = co2_emission_model(train_dataset.x)
yhat_drop = co2_emission_model_drop(train_dataset.x)

# plt.figure(figsize=(6.1, 7))
# plt.scatter(train_dataset.x.numpy(), train_dataset.y.numpy(), label="Samples")
# # plt.plot(data_set.x.numpy(), data_set.f.numpy(), label="True function", color='orange')
# plt.plot(train_dataset.x.numpy(), yhat.detach().numpy(), label='no dropout', c='r')
# plt.plot(train_dataset.x.numpy(), yhat_drop.detach().numpy(), label="dropout", c='g')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.xlim((-1, 1))
# plt.ylim((-2, 2.5))
# plt.legend(loc="best")
# plt.show()

# Plot the loss

plt.figure(figsize=(6.1, 7))
for key, value in training_results.items():
    plt.plot(np.log(np.array(value)), label=key)
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("Log of cost or total loss")
plt.show()

if __name__ == '__main__':
    print()
