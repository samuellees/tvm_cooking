import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

import torch

# no bias
# 
@auto_scheduler.register_workload
def sepconv(N, H, W, CO, CI, KH, KW, dtype):
    stride = 1
    groups = CI
    data = te.placeholder((N, CI, H, W), name="data", dtype=dtype)
    depth_kernel = te.placeholder((CI, CI // groups, KH, KW), name="depth_kernel", dtype=dtype)
    depth_conv = topi.nn.group_conv2d_nchw(data, depth_kernel, stride=stride, padding=0, dilation=1, groups=groups, out_dtype="float32")
    point_kernel = te.placeholder((CO, CI, 1, 1), name="point_kernel", dtype=dtype)
    out = topi.nn.conv2d_nchw(depth_conv, point_kernel, stride=1, padding=0, dilation=1, out_dtype="float32")
    return [data, depth_kernel, point_kernel, out]


# N, CI, H, W = (32, 32, 112, 112)
# CO, KH, KW = (64, 3, 3) 
# N, CI, H, W = (64, 3, 229, 229)
# CO, KH, KW = (10, 3, 3) 
N, CI, H, W = (3, 3, 20, 20)
CO, KH, KW = (10, 3, 3) 
target = tvm.target.Target("cuda")
log_file = "sepconv_test.json"
num_test_trails = 100
num_search_trails = 100

task = tvm.auto_scheduler.create_task(sepconv, (N, H, W, CO, CI, KH, KW, "float32"), target)
# print(task.compute_dag)

###
### do search
###
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=num_search_trails,
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
del measure_ctx

###
### use history
###
# inp, res = auto_scheduler.load_best(log_file, task.workload_key)
# sch, args = task.compute_dag.apply_steps_from_state(inp.state)


# print(tvm.lower(sch, args, simple_mode=True))

# torch tensor
data_torch = torch.rand((N, CI, H, W), dtype=torch.float)
depth_conv = torch.nn.Conv2d(CI, CI, (KH, KW), stride=1, padding=0, groups=CI, bias=False)
point_conv = torch.nn.Conv2d(CI, CO, (1, 1), stride=1, padding=0, groups=1, bias=False)
# np arrays
data_np = data_torch.numpy()
kernel_depth_np = depth_conv.weight.detach().numpy()
kernel_point_np = point_conv.weight.detach().numpy()
# torch result
data_torch = data_torch.cuda()
depth_conv = depth_conv.cuda()
point_conv = point_conv.cuda()
out = depth_conv(data_torch)
out = point_conv(out)
# np results
out_np = out.cpu().detach().numpy()

# torch.cuda.synchronize()
# time_start=time.time()
# for i in range(num_test_trails):
#     out = depth_conv(data_torch)
#     out = point_conv(out)
# torch.cuda.synchronize()
# time_end=time.time()
# print('avg time',(time_end-time_start) * 1000 / num_test_trails)

# tvm result
ctx = tvm.gpu()
func = tvm.build(sch, args, target)



data_tvm = tvm.nd.array(data_np, ctx=ctx) 
kernel_depth_tvm = tvm.nd.array(kernel_depth_np, ctx=ctx)
kernel_point_tvm = tvm.nd.array(kernel_point_np, ctx=ctx)
out_tvm = tvm.nd.empty(out_np.shape, ctx=ctx)

# start = time.time()
# for i in range(num_test_trails):
#     func(data_tvm, kernel_depth_tvm, kernel_point_tvm, out_tvm)
#     _ = out_tvm.asnumpy()
# end = time.time()
# start1 = time.time()
# for i in range(num_test_trails):
#     _ = out_tvm.asnumpy()
# end1 = time.time()
# print((end - start) - (end1 - start1)) * 1000 / num_test_trails)

# # Check results
np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

# # # Evaluate execution time.
# evaluator = func.time_evaluator(func.entry_name, ctx, repeat=num_test_trails, min_repeat_ms=500)
# print("Execution time of tvm operator: %.3f ms"
#     % (np.average(evaluator(data_tvm, kernel_depth_tvm, kernel_point_tvm, out_tvm).results) * 1000)
# )
