import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi


@auto_scheduler.register_workload
def pool_avg_fwd(N, C, H, W, KSIZE, stride, padding, dtype):
    data = te.placeholder((N, C, H, W), name="data", dtype=dtype)
    out = topi.nn.pool(data, [KSIZE, KSIZE], [stride, stride], [padding, padding, padding, padding], 
            "avg", ceil_mode=False, layout="NCHW", count_include_pad=True)
    return [data, out]


nchw_ksize_stride_pad_list = [
    [32, 1024, 7, 7, 7, 1, 0]
]
device = "cuda"
target = tvm.target.Target(device)
num_test_trails = 100
num_search_trails = 1000

for i in range(len(nchw_ksize_stride_pad_list)):

    time_begin = time.time()
    N, C, H, W, ksize, stride, padding = nchw_ksize_stride_pad_list[i]
    func_name = "pool_avg_fwd_n" + str(N) + "_c" + str(C) + "_h" + str(H) + "_w" + str(W)  \
         + "_k" + str(ksize) + "_s" + str(stride) + "_p" + str(padding) + "_" + str(device)
    log_file = func_name + ".json"
    
    # print(i+1)
    # print("%dx%dx%dx%d" % (N, C, H, W))
    # print("ksize=%d, stride=%d, pad=%d" % (ksize, stride, padding))
    # print(log_file)
    # print()

    task = tvm.auto_scheduler.create_task(pool_avg_fwd, (N, C, H, W, ksize, stride, padding, "float32"), target)

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
