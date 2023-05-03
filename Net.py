import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(D_in, H)
        self.output_layer = nn.Linear(H, D_out)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        hidden = self.hidden_layer(x)
        activated1 = self.sigmoid1(hidden)
        output = self.output_layer(activated1)
        activated2 = self.sigmoid2(output)
        return activated2


# make some data
# X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
# Y = torch.zeros(X.shape[0])
# Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0


# in most cases we might just have to pass a layer rather than input size etc

# Define a function to plot accuracy and loss

def plot_accuracy_loss(training_results):
    plt.subplot(2, 1, 1)
    plt.plot(training_results['training_loss'], 'r')
    plt.ylabel('loss')
    plt.title('training loss iterations')
    plt.subplot(2, 1, 2)
    plt.plot(training_results['validation_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.show()


def plot_accuracy_loss2(results):
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))

    ax[0, 0].plot(results['training data no dropout'], 'r')
    ax[0, 0].set_ylabel('loss')
    ax[0, 0].set_title('training data no dropout')

    ax[0, 1].plot(results['training data dropout'], 'r')
    ax[0, 1].set_ylabel('loss')
    ax[0, 1].set_title('training data dropout')

    ax[1, 0].plot(results['dropout accuracy'])
    ax[1, 0].set_title('dropout accuracy')
    ax[1, 0].set_ylabel('accuracy')
    ax[1, 0].set_xlabel('epochs')

    ax[1, 1].plot(results['no dropout accuracy'])
    ax[1, 1].set_title('no dropout accuracy')
    ax[1, 1].set_ylabel('accuracy')
    ax[1, 1].set_xlabel('epochs')

    plt.show()


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


class Data(Dataset):

    def __init__(self, Train=True):
        data = pd.read_csv('drug200.csv', delimiter=',')
        # pre-processing
        # drop any row with missing data
        data = data.dropna()

        X = data.drop(columns=['Drug']).values

        le_sex = LabelEncoder()
        le_sex.fit(['F', 'M'])
        X[:, 1] = le_sex.transform(X[:, 1])

        le_BP = LabelEncoder()
        le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
        X[:, 2] = le_BP.transform(X[:, 2])

        le_Chol = LabelEncoder()
        le_Chol.fit(['NORMAL', 'HIGH'])
        X[:, 3] = le_Chol.transform(X[:, 3])

        # normalize
        X = StandardScaler().fit_transform(X.astype(float))

        Y = data['Drug'].values
        le_Drug = LabelEncoder()
        le_Drug.fit(['drugX', 'drugY', 'drugC', 'drugA', 'drugB'])
        Y = le_Drug.transform(Y)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        self.x = torch.tensor(X_train if Train else X_test, dtype=torch.float32)
        self.y = torch.tensor(y_train if Train else y_test, dtype=torch.long)

        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


# def accuracy(model_, data_set):
#     _, yhat = torch.max(model_(data_set.x), 1)
#     return (yhat == data_set.y).numpy().mean()


drug_train_dataset = Data()
drug_validation_dataset = Data(Train=False)
layers = [5, 10, 5]
model = LayerNet(Layers=layers)
learning_rate = 0.10
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=drug_train_dataset, batch_size=32)
validation_loader = DataLoader(dataset=drug_validation_dataset, batch_size=32)
criterion = nn.CrossEntropyLoss()


def train(epochs):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            # loss for every iteration
            useful_stuff['training_loss'].append(loss.data.item())
        correct = 0
        for x, y in validation_loader:
            # validation
            z = model(x)
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
        accuracy = 100 * (correct / len(drug_validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    return useful_stuff


# training_results = train(epochs=30)
# Plot the accuracy and loss
# plot_accuracy_loss(training_results)

# it seems more hidden layers could cause overfitting, let's try dropout
model_drop = LayerNet(layers, p=0.5)
optimizer_drop = torch.optim.Adam(model_drop.parameters(), lr=0.01)


# we could try to see if dropout has any influence over the model's performance


def train_model(epochs):
    LOSS = {'training data no dropout': [], 'no dropout accuracy': [], 'training data dropout': [],
            'dropout accuracy': []}
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            yhat = model(x)
            yhat_drop = model_drop(x)
            loss = criterion(yhat, y)
            loss_drop = criterion(yhat_drop, y)
            optimizer.zero_grad()
            optimizer_drop.zero_grad()
            loss.backward()
            loss_drop.backward()
            optimizer.step()
            optimizer_drop.step()
            # store the loss for both models
            LOSS['training data no dropout'].append(loss.item())
            model_drop.eval()
            LOSS['training data dropout'].append(loss_drop.item())
            model_drop.train()

        correct = 0
        correct_drop = 0
        for i, (x, y) in enumerate(validation_loader):
            z = model(x)
            z_drop = model_drop(x)
            _, label = torch.max(z, 1)
            _, label_drop = torch.max(z_drop, 1)
            correct += (label == y).sum().item()
            correct_drop += (label_drop == y).sum().item()
            accuracy = 100 * (correct / len(drug_validation_dataset))
            model_drop.eval()
            accuracy_drop = 100 * (correct_drop / len(drug_validation_dataset))
            model_drop.train()
            LOSS['no dropout accuracy'].append(accuracy)
            LOSS['dropout accuracy'].append(accuracy_drop)
    return LOSS


training_results = train_model(epochs=70)
plot_accuracy_loss2(training_results)

# Print out the accuracy of the model without dropout
# print("The accuracy of the model without dropout: ", accuracy(model, drug_validation_dataset))
# print("The accuracy of the model with dropout: ", accuracy(model_drop, drug_validation_dataset))

# plt.figure(figsize=(6.1, 10))
#
#
# def plot_LOSS():
#     for key, value in LOSS.items():
#         plt.plot(np.log(np.array(value)), label=key)
#         plt.legend()
#         plt.xlabel("iterations")
#         plt.ylabel("Log of cost or total loss")
#
#
# # plot_LOSS()
# plt.show()
if __name__ == '__main__':
    print()
