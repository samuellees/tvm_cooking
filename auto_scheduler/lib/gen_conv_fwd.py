import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi


@auto_scheduler.register_workload
def conv_fwd(N, CI, H, W, CO, KSIZE, stride, padding, dtype):
    data = te.placeholder((N, CI, H, W), name="data", dtype=dtype)
    kernel = te.placeholder((CO, CI, KSIZE, KSIZE), name="kernel", dtype=dtype)
    out = topi.nn.conv2d_nchw(data, kernel, stride=stride, padding=padding, dilation=1, out_dtype="float32")
    return [data, kernel, out]





nchw_co_ksize_stride_pad_list = [
    # [32, 3, 5, 5, 2, 3, 1, 1],
    # [32, 3, 224, 224, 32, 3, 1, 1],
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
    [32, 512, 14, 14, 1024, 3, 2, 1]
]
device = "cuda"
target = tvm.target.Target(device)
num_test_trails = 100
num_search_trails = 1000

for i in range(len(nchw_co_ksize_stride_pad_list)):
    time_begin = time.time()
    N, CI, H, W, CO, ksize, stride, padding = nchw_co_ksize_stride_pad_list[i]
    # print(_+1)
    # print("%dx%dx%dx%d" % (N, CI, H, W))
    # print("CI=%d, CO=%d, ksize=%d, stride=%d, pad=%d" % (CI, CO, ksize, stride, padding))
    # print()
    func_name = "conv_fwd_n" + str(N) + "_c" + str(CI) + "_h" + str(H) + "_w" + str(W) + "_co"  \
         + str(CO) + "_k" + str(ksize) + "_s" + str(stride) + "_p" + str(padding) + "_" + str(device)
    log_file = func_name + ".json"

    task = tvm.auto_scheduler.create_task(conv_fwd, (N, CI, H, W, CO, ksize, stride, padding, "float32"), target)

    ### search
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_search_trails,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
    del measure_ctx

    ### load history
    # inp, res = auto_scheduler.load_best(log_file, task.workload_key)
    # sch, args = task.compute_dag.apply_steps_from_state(inp.state)

    # build func
    ctx = tvm.gpu()
    func = tvm.build(sch, args, target, name=func_name)
    # save result
    obj_fname = func_name + ".o"
    ptx_fname = func_name + ".ptx"
    func.save(obj_fname)
    func.imported_modules[0].save(ptx_fname)

    time_end = time.time()
    print("IterTime: ", (time_end - time_begin))
    exit()
