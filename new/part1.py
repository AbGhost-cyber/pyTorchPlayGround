import torch

u = torch.tensor([3.0, 5.0])
# print(torch.norm(u))
# print(torch.sqrt(sum(p.pow(2.0) for p in u)))
# print(torch.abs(u).sum())
a = torch.tensor(1.0, requires_grad=True)
f = 3 * a ** 2 - 4 * a
# f.backward()
# print(a.grad)

# How to reduce non-scalars to scalar for loss
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
y = torch.sum(x)
z = torch.tensor([[1., 0., -1.],
                  [-1., 0., 1.]])
# y.backward(z.sum())
# print(x.grad)

# how to exclude an intermediate term from the computational graph
x1 = torch.arange(4.0, requires_grad=True)
y1 = x1 * 2
u = y1.detach()
z1 = u * x1
z1.sum().backward()
# print(u)
# print(x1.grad)
a = torch.rand(size=(), requires_grad=True)
b = a * 2
# print(b.norm())

# for i in range(0, 2000, 32):
#     print(i)
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
my_slice = slice(2, 6)
if __name__ == '__main__':
    print()
