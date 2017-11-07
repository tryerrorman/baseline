import torch.nn as nn
from torch.autograd import Variable
import torch


def my_hook(grad):
    grad_clone = grad.clone()
    grad_clone = -1*grad_clone
    return grad_clone

x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3

h = z.register_hook(my_hook)

out = z.mean()


out.backward()


print(x.grad)


