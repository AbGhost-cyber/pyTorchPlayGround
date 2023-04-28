import torch
from torch.nn import Linear
import matplotlib.pyplot as plt

tensorA = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
tensorB = torch.tensor([[3, 1, 4, 1, 0], [15, 12, 6, 0, 0], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
# elem_wise = torch.mul(tensorA, tensorB)

newTensor1 = torch.tensor([[1, 1, 1], [1, 1, 1]])
newTensor2 = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
res = torch.mm(newTensor1, newTensor2)

# Create a 2D tensor
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Transpose the tensor
A_transposed = torch.transpose(A, 0, 1)

# Print the original and the transposed tensors
# print("Original tensor:")
# print(A)
# print("Transposed tensor:")
# print(A_transposed)

# Derivatives
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()

tensorX = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True)
y = 3 * tensorX ** 2 + 2 * tensorX + 1
y.backward(torch.ones(tensorX.size()))

# partial derivatives
x = torch.tensor([2.0], requires_grad=True)
y = (x + 5) ** 2
y.backward()
# dx = x.grad
# dy = y.grad
# print(f"partial derivative x: {dx}")
# print(f"partial derivative y: {dy}")

# playing around with linear regression
weight = torch.tensor(2.0, requires_grad=True)
bias = torch.tensor(-1.0, requires_grad=True)

# def forward(x):
#     yhat = weight * x + bias
#     return yhat


x1 = torch.tensor(2.0)
# yhat = forward(x1)
# print(yhat)

x1 = torch.tensor([[1.0], [2.0]])
# print("The shape of x: ", x1.size())

linear_model = Linear(in_features=1, out_features=1, bias=True)
# print(f"params: {list(linear_model.parameters())}")
# print(linear_model.state_dict())

# Create f(X) with a slope of 1 and a bias of -3
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 3
Y = f + 0.1 * torch.randn(X.size())


# plt.plot(X.numpy(), Y.numpy(), 'rx', label='y')
# plt.plot(X.numpy(), f.numpy(), label='f')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()


def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)


def forward(x):
    yhat = w * x + b
    return yhat


# train the model
w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)

lr = 0.3
LOSS = []


def train_model(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        loss = criterion(Yhat, Y)
        LOSS.append(loss.detach())
        loss.backward()
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()


train_model(15)
plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.show()

if __name__ == '__main__':
    print()
