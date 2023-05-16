import torch
import torch.nn as nn

conv = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
print(conv.weight.shape, conv.bias.shape)

if __name__ == '__main__':
    print()
