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


class LayerNet(nn.Module):
    def __init__(self, Layers):
        super(LayerNet, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, activation):
        length = len(self.hidden)
        for index, linear_transform in zip(range(length), self.hidden):
            if index < length - 1:
                activation = F.relu(linear_transform(activation))
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
        self.y = torch.tensor(y_train if Train else y_test, dtype=torch.float32)

        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


# def accuracy(model_, data_set):
#     _, yhat = toe

drug_train_dataset = Data()
drug_validation_dataset = Data(Train=False)
layers = [5, 10, 10, 5]
model = LayerNet(Layers=layers)
learning_rate = 0.10
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=drug_train_dataset, batch_size=32)
validation_loader = DataLoader(dataset=drug_validation_dataset, batch_size=32)
criterion = nn.CrossEntropyLoss()

print(model.state_dict())
print(drug_train_dataset.x.shape)


def train(epochs):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y.long())
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


#
# no, i mean it doesn't compile, RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x5 and 2x10)

training_results = train(epochs=50)
# Plot the accuracy and loss
plot_accuracy_loss(training_results)

if __name__ == '__main__':
    print()
