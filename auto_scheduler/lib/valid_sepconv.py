import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

import torch

# no bias
# 
@auto_scheduler.register_workload
def sepconv(N, H, W, CO, CI, KH, KW, stride, dtype):
    groups = CI
    data = te.placeholder((N, CI, H, W), name="data", dtype=dtype)
    depth_kernel = te.placeholder((CI, CI // groups, KH, KW), name="depth_kernel", dtype=dtype)
    depth_conv = topi.nn.group_conv2d_nchw(data, depth_kernel, stride, padding=1, dilation=1, groups=groups, out_dtype="float32")
    point_kernel = te.placeholder((CO, CI, 1, 1), name="point_kernel", dtype=dtype)
    out = topi.nn.conv2d_nchw(depth_conv, point_kernel, stride=1, padding=0, dilation=1, out_dtype="float32")
    return [data, depth_kernel, point_kernel, out]


nchw_list = [[32, 32, 112, 112],
             [32, 64, 112, 112],
             [32, 128, 56, 56],
             [32, 128, 56, 56],
             [32, 256, 28, 28],
             [32, 256, 28, 28],
             [32, 512, 14, 14],
             [32, 512, 14, 14],
             [32, 1024, 7, 7],
             [32, 1024, 7, 7]]
co_list = [64, 
           128,
           128,
           256,
           256,
           512,
           512,
           1024,
           1024]
stride_list = [1, 2, 1, 2, 1, 2, 1, 2, 1]
target = tvm.target.Target("cuda")

for i in range(len(co_list)):
    num_test_trails = 100
    nchw = nchw_list[i]
    stride = stride_list[i]
    N, CI, H, W = nchw
    CO, KH, KW = (co_list[i], 3, 3) 
    target = tvm.target.Target("cuda")
    func_name = "sepconv_" + str(nchw[1]) + "_" + str(co_list[i]) + "_1_" + str(stride)
    log_file = func_name + ".json"

    task = tvm.auto_scheduler.create_task(sepconv, (N, H, W, CO, CI, KH, KW, stride, "float32"), target)
    inp, res = auto_scheduler.load_best(log_file, task.workload_key)
    sch, args = task.compute_dag.apply_steps_from_state(inp.state)

    # torch tensor
    data_torch = torch.rand((N, CI, H, W), dtype=torch.float)
    depth_conv = torch.nn.Conv2d(CI, CI, (KH, KW), stride=stride, padding=1, groups=CI, bias=False)
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

    # tvm result
    ctx = tvm.gpu()
    func = tvm.build(sch, args, target, name=func_name)
    data_tvm = tvm.nd.array(data_np, ctx=ctx) 
    kernel_depth_tvm = tvm.nd.array(kernel_depth_np, ctx=ctx)
    kernel_point_tvm = tvm.nd.array(kernel_point_np, ctx=ctx)
    out_tvm = tvm.nd.empty(out_np.shape, ctx=ctx)
    func(data_tvm, kernel_depth_tvm, kernel_point_tvm, out_tvm)

    # # Check results
    # np.testing.assert_allclose(data_np, data_tvm.asnumpy(), rtol=1e-3)
    # np.testing.assert_allclose(kernel_depth_np, kernel_depth_tvm.asnumpy(), rtol=1e-3)
    # np.testing.assert_allclose(kernel_point_np, kernel_point_tvm.asnumpy(), rtol=1e-3)
    # np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

    # torch timing
    torch.cuda.synchronize()
    time_start=time.time()
    for _ in range(num_test_trails):
        out = depth_conv(data_torch)
        out = point_conv(out)
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
        func(data_tvm, kernel_depth_tvm, kernel_point_tvm, out_tvm)
    _ = out_tvm.asnumpy()
    end = time.time()
    _ = out_tvm.asnumpy()
    end2 = time.time()
    print("%d\'th layer: tvm time: %.3f"
            % (i, (end - start - (end2 - end)) * 1000 / num_test_trails))
    print("")
