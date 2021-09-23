import torch
import torch_struct

import time

def timeit(f, iters=10):
    start_time = time.time()
    for i in range(iters):
        f()
    end_time = time.time()
    dur = end_time - start_time
    print(f"{dur}s")
    return dur

T = 4
X = 32
#T = 16
#X = 256

grad_output = torch.randn(T, X, X)
log_potentials = torch.randn(T, X, X)
log_potentials.requires_grad_(True)

def A(potentials):
    T, X, _ = potentials.shape
    A = torch.zeros(X)
    for t in range(T):
        A = (A[:,None] + potentials[t]).logsumexp(0)
    return A.logsumexp(0)


struct = torch_struct.LinearChainCRF(log_potentials[None].transpose(-1,-2))
lp = struct.partition
m = struct.marginals.transpose(-1, -2)

# slow, instantiate full hessian
def slow():
    hessian = torch.autograd.functional.hessian(A, log_potentials)
    vhp = torch.einsum("abc,abcdef->def", grad_output, hessian)
    return vhp

# faster, i believe forward on reverse? need to check:
# https://pytorch.org/docs/stable/_modules/torch/autograd/functional.html#vhp
def faster():
    _, vhp = torch.autograd.functional.vhp(A, log_potentials, grad_output)
    return vhp

def manual():
    log_partition = A(log_potentials)
    marginals, = torch.autograd.grad(log_partition, log_potentials, retain_graph=True, create_graph=True)
    vhp, = torch.autograd.grad(marginals, log_potentials, grad_output)
    return vhp

print(torch.allclose(faster(), manual()))

#timeit(slow) # 15.5s
timeit(faster) # .0075s
timeit(manual) # .0075s

#H = torch.autograd.grad(m, log_potentials)
import pdb; pdb.set_trace()
