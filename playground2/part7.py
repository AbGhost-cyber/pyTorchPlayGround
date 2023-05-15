import torch

torch.manual_seed(1)
tensorA = torch.rand((3, 2, 2))
print(tensorA.view(3, -1))
print(tensorA.view(3, -1).mean(dim=))
if __name__ == '__main__':
    print()
