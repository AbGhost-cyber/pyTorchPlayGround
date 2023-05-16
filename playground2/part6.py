import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

cifar10 = datasets.CIFAR10('data', train=True, download=True)
cifar10_val = datasets.CIFAR10('data', train=False, download=True)

to_tensor = transforms.ToTensor()
pic, label = cifar10[99]
img_t = to_tensor(pic)

# we can also pass the transform directly as an argument to dataset
tensor_cifar10 = datasets.CIFAR10('data', train=True, download=False, transform=transforms.ToTensor())

# the ToTensor transform turns the data into a 32-bit floating-point per channel,
# scaling the values down from 0.0 to 1.0
# print(img_t.min(), img_t.max())

# compute for mean and std deviation, dim is where to concatenate
imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
# mean = imgs.view(3, -1).mean(dim=1)
# std = imgs.view(3, -1).std(dim=1)

# with the numbers gotten we can normalize
transformed_cifar10 = datasets.CIFAR10('data', train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
]))
transformed_cifar10_val = datasets.CIFAR10('data', train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
]))
# img_t, _ = transformed_cifar10[99]
# plt.imshow(img_t.permute(1, 2, 0))
# plt.show()

# filter for bird and airplanes
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in transformed_cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in transformed_cifar10_val if label in [0, 2]]

# we can start building
n_out = 2
model = nn.Sequential(nn.Linear(3072, 512),
                      nn.Tanh(),
                      nn.Linear(512, n_out),
                      nn.LogSoftmax(dim=1))
# since our model expects 3072 features in the input, we need to transform into 1D tensor
img, _ = cifar2[0]
# flatten and add an extra dimension at the beginning of the tensor, effectively creating a 2D tensor with a single row
# img_batch = img.view(-1).unsqueeze(0)
# out = model(img_batch)
# print(out)
# _, index = torch.max(out, dim=1)
# print(index)
# instantiate our NLL loss
loss_fn = nn.NLLLoss()
# print(loss(out, torch.tensor([label])))
learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
n_epochs = 100
# for epoch in range(n_epochs):
#     for img, label in cifar2:
#         out = model(img.view(-1).unsqueeze(0))
#         loss = loss_fn(out, torch.tensor([label]))
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

# the above code takes too much time, we can try data loader
train_loader = DataLoader(cifar2, batch_size=64, shuffle=True)
# for epoch in range(n_epochs):
#     for imgs, labels in train_loader:
#         batch_size = imgs.shape[0]
#         outputs = model(imgs.view(batch_size, -1))
#         loss = loss_fn(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

# validate
val_loader = DataLoader(cifar2_val, batch_size=64, shuffle=False)
correct = 0
total = 0
# with torch.no_grad():
#     for imgs, labels in val_loader:
#         batch_size = imgs.shape[0]
#         outputs = model(imgs.view(batch_size, -1))
#         _, predicted = torch.max(outputs, dim=1)
#         total += labels.shape[0]
#         correct += int((predicted == labels).sum())
#     print("Accuracy: %f", correct / total)
#     print("total", total)

# make our model more complicated by adding extra hidden layers
model = nn.Sequential(
    nn.Linear(3072, 1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 128),
    nn.Tanh(),
    nn.Linear(128, 2),
    nn.LogSoftmax(dim=1))
# note: we can drop the LogSoftmax and nn.NLLLOSS for CrossEntropyLoss, it's the same
# numel_list = [p.numel() for p in model.parameters() if p.requires_grad]
# print(sum(numel_list), numel_list)

# using convolution
conv = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
# print(conv.weight.shape, conv.bias.shape)
img, _ = cifar2[0]
# we need to add the zeroth batch dimension with unsqueeze if we want to call the conv module
# with one input image, since nn.Conv2d expects a B × C × H × W shaped tensor as input:
output = conv(img.unsqueeze(0))


# we can play with convolution by setting weights by hand and see what happens.
# with torch.no_grad():
#     conv.bias.zero_()
# with torch.no_grad():
#     conv.weight.fill_(1.0 / 9.0)
# plot to see
# output = conv(img.unsqueeze(0))
# plt.imshow(output[0, 0].detach(), cmap='gray')
# plt.show()  # result is blurry
# let's see another
# conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
# with torch.no_grad():
#     conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
#                                    [-1.0, 0.0, 1.0],
#                                    [-1.0, 0.0, 1.0]])
#     conv.bias.zero_()
# using max pooling
# pool = nn.MaxPool2d(2)  # down sample by half
# output = pool(img.unsqueeze(0))
# print(output.shape)

# model = nn.Sequential(
#     nn.Conv2d(3, 16, kernel_size=3, padding=1),
#     nn.Tanh(),
#     nn.MaxPool2d(2),
#     nn.Conv2d(16, 8, kernel_size=3, padding=1),
#     nn.Tanh(),
#     nn.MaxPool2d(2)
# )

# creating our own network as an nn.Module

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(in_features=8 * 8 * 8, out_features=32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        # First convolutional layer
        out = self.pool1(self.ac1(self.conv1(x)))
        # Second convolutional layer
        out = self.pool2(self.act2(self.conv2(x)))
        # Flatten output
        out = out.view(-1, 8 * 8 * 8)
        # Fully connected layer 1
        out = self.act3(self.fc1(out))
        # Fully connected layer 2
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    print()
