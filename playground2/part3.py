import os

import imageio as imageio
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models, transforms
from torch import hub
import pandas as pd

torch.manual_seed(1)
# tensor playground

# resnet_18_model = hub.load('pytorch/vision:master', 'resnet18', pretrained=True)

img_t = torch.randn(3, 5, 5)
weights = torch.tensor([0.2126, 0.7152, 0.0722])
batch_t = torch.randn(2, 3, 5, 5)

# lazy and unweighted mean
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)

unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(1)
img_weights = (img_t * unsqueezed_weights)
batch_weights = (batch_t * unsqueezed_weights)
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)

# the above code is error-prone
# below code is experimental
# weight_names = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
# img_named = img_t.refine_names('channels', 'rows', 'columns')
# print(img_named.names)
# points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
# points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# points_gpu = points.cuda()
# print(torch.__config__.show())

# img_arr = imageio.v2.imread("/Users/mac/Downloads/cat.jpeg")
# img = torch.from_numpy(img_arr)
# out = img.permute(2, 0, 1)

data_dir = '/Users/mac/Downloads/image-cats/'
filenames = [name for name in os.listdir(data_dir)]
batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

# for i, filename in enumerate(filenames):
#     img_arr = imageio.v2.imread(os.path.join(data_dir, filename))
#     img_t = torch.from_numpy(img_arr)
#     img_t = img_t.permute(2, 0, 1)
#     img_t = img_t[:3]
#     batch[i] = img_t

# next we normalize
# batch = batch.float()
# batch /= 255.0

# we could also:
# n_channels = batch.shape[1]
# for c in range(n_channels):
#     mean = torch.mean(batch[:, c])
#     std = torch.std(batch[:, c])
#     batch[:, c] = (batch[:, c] - mean) / std

# using pandas
data_frame = pd.read_csv('/Users/mac/Downloads/winequality-white.csv', delimiter=';')
wine_features = data_frame.drop(columns=['quality']).values
wine_labels = data_frame['quality'].values

X = torch.tensor(wine_features, dtype=torch.float32)
Y = torch.tensor(wine_labels, dtype=torch.long)

# one hot encoding
Y_onehot = torch.zeros(Y.shape[0], 10)
Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

# obtain mean and std of each column of our feature
data_mean = torch.mean(X, dim=0)
data_std = torch.std(X, dim=0)
# normalize
normalized_data = (X - data_mean) / data_std

# let's use our eyes to tell good and bad wines apart
bad_indexes = Y <= 3
bad_data = X[bad_indexes]
mid_data = X[(Y > 3) & (Y < 7)]
good_data = X[Y >= 7]
bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(data_frame.columns, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

# a threshold on total sulfur dioxide as a crude criterion for discriminating good wines from bad ones
total_sulfur_threshold = 141.83
total_sulfur_data = X[:, 6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)

# good wines
actual_indexes = Y > 5


if __name__ == '__main__':
    print()
