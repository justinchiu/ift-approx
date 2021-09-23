import torch

Z = 64

theta = torch.randn(Z)
theta.requires_grad_(True)

# ACCURACY of gradient

z = theta.softmax(-1)
v = torch.randn(Z)
zbar, = torch.autograd.grad(z, theta, v)

zbar1 = z * (v - v.dot(z))

zbar_tilde_1 = -v * (2 + 1/z)
# looks like 1st order approximation is horrible
print("Error of first order Neumann VJP approximation")
print((zbar - zbar_tilde_1).abs().max())
import pdb; pdb.set_trace()


