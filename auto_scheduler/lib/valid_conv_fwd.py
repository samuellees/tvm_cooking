import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

import torch

nchw_co_ksize_stride_pad_list = [
    [32, 3, 224, 224, 32, 3, 1, 1],
    # [32, 32, 224, 224, 64, 3, 2, 1],
    # [32, 64, 112, 112, 64, 3, 1, 1],

    # [32, 64, 112, 112, 128, 3, 2, 1],
    # [32, 128, 56, 56, 128, 3, 1, 1],
    # [32, 64, 112, 112, 128, 1, 2, 0],
    # [32, 128, 56, 56, 256, 3, 2, 1],
    # [32, 256, 28, 28, 256, 3, 1, 1],
    # [32, 128, 56, 56, 256, 1, 2, 0],
    # [32, 256, 28, 28, 512, 3, 2, 1],
    # [32, 512, 14, 14, 512, 3, 1, 1],
    # [32, 256, 28, 28, 512, 1, 2, 0],
    # [32, 512, 14, 14, 1024, 3, 2, 1]
]
device = "cuda"
target = tvm.target.Target(device)
num_test_trails = 1

for i in range(len(nchw_co_ksize_stride_pad_list)):
    time_begin = time.time()
    N, CI, H, W, CO, ksize, stride, padding = nchw_co_ksize_stride_pad_list[i]
    
    func_name = "conv_fwd_n" + str(N) + "_c" + str(CI) + "_h" + str(H) + "_w" + str(W) + "_co"  \
         + str(CO) + "_k" + str(ksize) + "_s" + str(stride) + "_p" + str(padding) + "_" + str(device)
    print(func_name)
    
    log_file = func_name + ".json"

    func_module = tvm.runtime.load_module(func_name + ".o")
    func_module_dev = tvm.runtime.load_module(func_name + ".ptx")
    func_module.import_module(func_module_dev)
    func = func_module.get_function(func_name)

    # torch tensor
    data_torch = torch.rand((N, CI, H, W), dtype=torch.float)
    conv_torch = torch.nn.Conv2d(CI, CO, (ksize, ksize), stride=stride, padding=padding, groups=1, bias=False)
    # np arrays
    data_np = data_torch.numpy()
    kernel_np = conv_torch.weight.detach().numpy()
    # torch result
    data_torch = data_torch.cuda()
    conv_torch = conv_torch.cuda()
    out = conv_torch(data_torch)
    # np results
    out_np = out.cpu().detach().numpy()

    # tvm result
    ctx = tvm.gpu()
    data_tvm = tvm.nd.array(data_np, ctx=ctx) 
    kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
    out_tvm = tvm.nd.empty(out_np.shape, ctx=ctx)
    func(data_tvm, kernel_tvm, out_tvm)

    # # Check results
    # np.testing.assert_allclose(data_np, data_tvm.asnumpy(), rtol=1e-3)
    # np.testing.assert_allclose(kernel_np, kernel_tvm.asnumpy(), rtol=1e-3)
    np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

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
    _ = out_tvm.asnumpy()
    end = time.time()
    _ = out_tvm.asnumpy()
    end2 = time.time()
    print("%d\'th layer: tvm time: %.3f"
            % (i, (end - start - (end2 - end)) * 1000 / num_test_trails))
    print("")
