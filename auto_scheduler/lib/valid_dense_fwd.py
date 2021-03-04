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
N, CI, CO = 32, 1024, 10

time_begin = time.time()
target = tvm.target.Target("cuda")
func_name = "dense_fwd_n" + str(N) + "_ci" + str(CI) + "_co" + str(CO) + "_" + "cuda"
log_file = func_name + ".json"

func_module = tvm.runtime.load_module(func_name + ".o")
func_module_dev = tvm.runtime.load_module(func_name + ".ptx")
func_module.import_module(func_module_dev)
func = func_module.get_function(func_name)

# torch tensor
dshape = (N, CI)
wshape = (CO, CI)
bshape = (CO, )
data_torch = torch.rand(dshape, dtype=torch.float, requires_grad=True).cuda()
weight_torch = torch.rand(wshape, dtype=torch.float, requires_grad=True).cuda()
bias_torch = torch.rand(bshape, dtype=torch.float, requires_grad=True).cuda()
out_torch = F.linear(data_torch, weight_torch, bias_torch)
# np arrays
data_np = data_torch.cpu().detach().numpy()
weight_np = weight_torch.cpu().detach().numpy()
bias_np = bias_torch.cpu().detach().numpy()
out_np = out_torch.cpu().detach().numpy()

# tvm result
ctx = tvm.gpu()
data_tvm = tvm.nd.array(data_np, ctx=ctx) 
weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
bias_tvm = tvm.nd.array(bias_np, ctx=ctx)
output_tvm = tvm.nd.empty(out_torch.shape, ctx=ctx)
func(data_tvm, weight_tvm, bias_tvm, output_tvm)

# # Check results
np.testing.assert_allclose(data_np, data_tvm.asnumpy(), rtol=1e-3)
np.testing.assert_allclose(weight_np, weight_tvm.asnumpy(), rtol=1e-3)
np.testing.assert_allclose(bias_np, bias_tvm.asnumpy(), rtol=1e-3)
np.testing.assert_allclose(out_np, output_tvm.asnumpy(), rtol=1e-3)
# np.testing.assert_allclose(grad_input_np, grad_input_tvm.asnumpy(), rtol=1e-3)
# np.testing.assert_allclose(grad_weight_np, grad_weight_tvm.asnumpy(), rtol=1e-3)

# print(grad_input_np)
# print(grad_input_tvm)

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