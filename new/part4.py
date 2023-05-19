import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils


def show_images(imgs, titles=None):
    """Plot a list of images."""
    img_grid = utils.make_grid(imgs)
    img_grid = img_grid.permute(1, 2, 0)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_grid)
    rows, cols = 1, 8
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            label = "hi" # change later lol
            ax.text(j / cols, 1 + i / rows, label, ha='center', va='center', transform=ax.transAxes, fontsize=13)
    plt.axis('off')
    plt.show()


class FashionMNIST(Dataset):
    def __init__(self, batch_size=8, resize=(28, 28), num_workers=0):
        self.batch_size = batch_size
        transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.train = datasets.FashionMNIST(root='fashion', train=True, download=True, transform=transform)
        self.val = datasets.FashionMNIST(root='fashion', train=False, download=True, transform=transform)
        self.num_workers = num_workers

    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return DataLoader(dataset=data, batch_size=self.batch_size, shuffle=train, num_workers=self.num_workers)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def visualize(self, batch, labels=None):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(0), titles=labels)


data = FashionMNIST()
batch = next(iter(data.train_dataloader()))
data.visualize(batch=batch)
# X, y = next(iter(data.train_dataloader()))
# print(X.shape, X.dtype, y.shape, y.dtype)
# tic = time.time()
# for X, y in data.train_dataloader():
#     continue
# print(f'{time.time() - tic:.2f} sec')
if __name__ == '__main__':
    print()
