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
N, C, H, W = 8, 3, 10, 10

time_begin = time.time()
target = tvm.target.Target("cuda")
func_name = "batch_norm_bwd_n" + str(N) + "_c" + str(C) + "_h" + str(H) + "_w" + str(W) + "_" + "cuda"
log_file = func_name + ".json"

func_module = tvm.runtime.load_module(func_name + ".o")
func_module_dev = tvm.runtime.load_module(func_name + ".ptx")
func_module.import_module(func_module_dev)
func = func_module.get_function(func_name)

# torch tensor
dshape = (N, C, H, W)
oshape = (C, )
bshape = (1, C, 1, 1)
sshape = (1, )
data_torch = torch.rand(dshape, dtype=torch.float, requires_grad=True).cuda()
data_torch.retain_grad()
eps_torch = torch.rand(sshape, dtype=torch.float)
batch_norm = torch.nn.BatchNorm2d(dshape[1], eps_torch.numpy()[0], affine=True, track_running_stats=True).cuda()
# get internal tensor
running_mean_torch = batch_norm.running_mean
running_var_torch = batch_norm.running_var
scale_torch = batch_norm.weight
bias_torch = batch_norm.bias
# init
running_mean_torch.data.fill_(0.0)
running_var_torch.data.fill_(1.0)
scale_torch.data.normal_()
bias_torch.data.normal_()
# comput output
out_torch = batch_norm(data_torch)
grad_output_torch = torch.rand(dshape, dtype=torch.float).cuda()
out_torch.backward(grad_output_torch)
# np arrays input
data_np = data_torch.cpu().detach().numpy()
scale_np = scale_torch.cpu().detach().numpy()
eps_np = eps_torch.cpu().detach().numpy()
saved_mean_np = data_np.mean((0, 2, 3))
saved_var_np = np.sqrt(data_np.var((0, 2, 3)) + eps_np)
grad_output_np = grad_output_torch.cpu().detach().numpy()
# np arrays output
grad_input_np = data_torch.grad.cpu().detach().numpy()
grad_scale_np = scale_torch.grad.cpu().detach().numpy()
grad_bias_np = bias_torch.grad.cpu().detach().numpy()

# tvm input
ctx = tvm.gpu()
data_tvm = tvm.nd.array(data_np, ctx=ctx)
scale_tvm = tvm.nd.array(scale_np, ctx=ctx)
saved_mean_tvm = tvm.nd.array(saved_mean_np, ctx=ctx)
saved_var_tvm = tvm.nd.array(saved_var_np, ctx=ctx)
eps_tvm = tvm.nd.array(eps_np, ctx=ctx)
grad_output_tvm = tvm.nd.array(grad_output_np, ctx=ctx)
# tvm output
grad_input_tvm = tvm.nd.empty(grad_input_np.shape, ctx=ctx)
grad_scale_tvm = tvm.nd.empty(grad_scale_np.shape, ctx=ctx)
grad_bias_tvm = tvm.nd.empty(grad_bias_np.shape, ctx=ctx)
func(data_tvm, scale_tvm, saved_mean_tvm, saved_var_tvm, eps_tvm, grad_output_tvm, grad_input_tvm, grad_scale_tvm, grad_bias_tvm)
# Check results
np.testing.assert_allclose(data_np, data_tvm.asnumpy(), atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(scale_np, scale_tvm.asnumpy(), atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(saved_mean_np, saved_mean_tvm.asnumpy(), atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(saved_var_np, saved_var_tvm.asnumpy(), atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(grad_output_np, grad_output_tvm.asnumpy(), atol=1e-3, rtol=1e-3)

np.testing.assert_allclose(grad_input_np, grad_input_tvm.asnumpy(), atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(grad_scale_np, grad_scale_tvm.asnumpy(), atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(grad_bias_np, grad_bias_tvm.asnumpy(), atol=1e-3, rtol=1e-3)

print(grad_scale_np, grad_scale_tvm.asnumpy())

'''
# torch timing
torch.cuda.synchronize()
time_start=time.time()
for _ in range(num_test_trails):
    out = conv_torch(data_torch)
    torch.cuda.synchronize()
time_end=time.time()
print('%d\'th layer: torch time: %.3f' % (i, (time_end-time_start) * 1000.0 / num_test_trails))

# tvm timing.
# evaluator = func.time_evaluator(func.entry_name, ctx, repeat=num_test_trails, min_repeat_ms=500)
# print("%d\'th layer: tvm time: %.3f"
#     % (i, np.average(evaluator(data_tvm, kernel_depth_tvm, kernel_point_tvm, out_tvm).results) * 1000)
# )

start = time.time()
for _ in range(num_test_trails):
    func(data_tvm, kernel_tvm, out_tvm)
    
out_tvm.ctx.sync()
# _ = out_tvm.asnumpy()
end = time.time()
# _ = out_tvm.asnumpy()
end2 = time.time()
print("%d\'th layer: tvm time: %.3f"
        % (i, (end - start - (end2 - end)) * 1000 / num_test_trails))
print("")
'''