import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

import torch
import torch.nn.functional as F


device = "cuda"
target = tvm.target.Target(device)
num_test_trails = 100

time_begin = time.time()
N, C, H, W = 32, 32, 224, 224

time_begin = time.time()
target = tvm.target.Target("cuda")
func_name = "relu_bwd_n" + str(N) + "_c" + str(C) + "_h" + str(H) + "_w" + str(W) + "_" + str(device)
log_file = func_name + ".json"

func_module = tvm.runtime.load_module(func_name + ".o")
func_module_dev = tvm.runtime.load_module(func_name + ".ptx")
func_module.import_module(func_module_dev)
func = func_module.get_function(func_name)

# torch tensor
dshape = (N, C, H, W)
data_torch = torch.rand(dshape, dtype=torch.float, requires_grad=True).cuda()
data_torch.retain_grad()
relu = torch.nn.ReLU().cuda()
out_torch = relu(data_torch)
grad_output_torch = torch.ones(dshape, dtype=torch.float).cuda()
out_torch.backward(grad_output_torch)
# np arrays
data_np = data_torch.cpu().detach().numpy()
grad_output_np = grad_output_torch.cpu().detach().numpy()
grad_input_np = data_torch.grad.cpu().detach().numpy()

# tvm result
ctx = tvm.gpu()
data_tvm = tvm.nd.array(data_np, ctx=ctx) 
grad_output_tvm = tvm.nd.array(grad_output_np, ctx=ctx)
grad_input_tvm = tvm.nd.empty(data_torch.shape, ctx=ctx)
func(data_tvm, grad_output_tvm, grad_input_tvm)

# # Check results
np.testing.assert_allclose(data_np, data_tvm.asnumpy(), rtol=1e-3)
np.testing.assert_allclose(grad_output_np, grad_output_tvm.asnumpy(), rtol=1e-3)
np.testing.assert_allclose(grad_input_np, grad_input_tvm.asnumpy(), rtol=1e-3)