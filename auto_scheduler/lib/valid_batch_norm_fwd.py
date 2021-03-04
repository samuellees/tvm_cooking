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
N, C, H, W = 10, 10, 10, 10
eps = 1e-5
momentum = 0.1

time_begin = time.time()
target = tvm.target.Target("cuda")
func_name = "batch_norm_fwd_n" + str(N) + "_c" + str(C) + "_h" + str(H) + "_w" + str(W) + "_" + "cuda"
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
eps_torch = torch.rand(sshape, dtype=torch.float)
momentum_torch = torch.rand(sshape, dtype=torch.float)
batch_norm = torch.nn.BatchNorm2d(dshape[1], eps_torch.numpy()[0], momentum_torch.numpy()[0], affine=True, track_running_stats=True).cuda()
# get internal tensor
running_mean_torch = batch_norm.running_mean
running_var_torch = batch_norm.running_var
scale_torch = batch_norm.weight
bias_torch = batch_norm.bias
# init
running_mean_torch.data.normal_()
running_var_torch.data.normal_()
scale_torch.data.normal_()
bias_torch.data.normal_()
# np arrays input
data_np = data_torch.cpu().detach().numpy()
scale_np = scale_torch.cpu().detach().numpy()
bias_np = bias_torch.cpu().detach().numpy()
running_mean_np = running_mean_torch.cpu().detach().numpy()
running_var_np = running_var_torch.cpu().detach().numpy()
eps_np = eps_torch.cpu().detach().numpy()
momentum_np = momentum_torch.cpu().detach().numpy()

ctx = tvm.gpu()
data_tvm = tvm.nd.array(data_np, ctx=ctx)
scale_tvm = tvm.nd.array(scale_np, ctx=ctx)
bias_tvm = tvm.nd.array(bias_np, ctx=ctx)
running_mean_tvm = tvm.nd.array(running_mean_np, ctx=ctx)
running_var_tvm = tvm.nd.array(running_var_np, ctx=ctx)
eps_tvm = tvm.nd.array(eps_np, ctx=ctx)
momentum_tvm = tvm.nd.array(momentum_np, ctx=ctx)
# comput output
out_torch = batch_norm(data_torch)
# np arrays output
out_np = out_torch.cpu().detach().numpy()
running_mean_np = running_mean_torch.cpu().detach().numpy()
running_var_np = running_var_torch.cpu().detach().numpy()
# tvm result
out_tvm = tvm.nd.empty(out_torch.shape, ctx=ctx)
saved_mean_tvm = tvm.nd.empty(running_mean_torch.shape, ctx=ctx)
saved_rvars_tvm = tvm.nd.empty(running_mean_torch.shape, ctx=ctx)
running_mean_tvm2 = tvm.nd.empty(running_mean_np.shape, ctx=ctx)
running_var_tvm2 = tvm.nd.empty(running_var_np.shape, ctx=ctx)
func(data_tvm, scale_tvm, bias_tvm, running_mean_tvm, running_var_tvm, 
    momentum_tvm, eps_tvm, out_tvm, saved_mean_tvm, saved_rvars_tvm, running_mean_tvm2, running_var_tvm2)
# Check results
np.testing.assert_allclose(data_np, data_tvm.asnumpy(), rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(scale_np, scale_tvm.asnumpy(), rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(bias_np, bias_tvm.asnumpy(), rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(running_mean_np, running_mean_tvm2.asnumpy(), rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(running_var_np, running_var_tvm2.asnumpy(), rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(data_np.mean((0, 2, 3)), saved_mean_tvm.asnumpy(), rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(np.sqrt(data_np.var((0, 2, 3)) + eps_np), saved_rvars_tvm.asnumpy(), rtol=1e-3, atol=1e-3)

# print(data_np.mean((0, 2, 3)))
# print(saved_mean_tvm.asnumpy())

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