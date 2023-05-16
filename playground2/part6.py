import matplotlib.pyplot as plt
import torch
from torch import nn, optim
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
for epoch in range(n_epochs):
    for img, label in cifar2:
        out = model(img.view(-1).unsqueeze(0))
        loss = loss_fn(out, torch.tensor([label]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

if __name__ == '__main__':
    print()
