import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

@auto_scheduler.register_workload
def dense_bwd(N, CI, CO, dtype):
    data = te.placeholder((N, CI), name="data", dtype=dtype)
    weight = te.placeholder((CO, CI), name="weight", dtype=dtype)
    grad_out = te.placeholder((N, CO), name="grad_out", dtype=dtype)
    k = te.reduce_axis((0, CO), name="k")
    grad_data = te.compute(
        (N, CI),
        lambda i, j: te.sum(grad_out[i, k] * weight[k, j], axis=k),
        name="grad_data",
        tag="dense_bwd_data",
    )

    kk = te.reduce_axis((0, N), name="kk")
    grad_weight = te.compute(
        (CO, CI),
        lambda i, j: te.sum(grad_out[kk, i] * data[kk, j], axis=kk),
        name="grad_weight",
        tag="dense_bwd_weight",
    )

    return [data, weight, grad_out, grad_data, grad_weight]


n_ci_co_list = [
    # [32, 1024 * 7 * 7, 1024],
    [32, 1024, 10]
]
device = "cuda"
target = tvm.target.Target(device)
num_test_trails = 100
num_search_trails = 1000

for i in range(len(n_ci_co_list)):

    time_begin = time.time()
    N, CI, CO = n_ci_co_list[i]
    func_name = "dense_bwd_n" + str(N) + "_ci" + str(CI) + "_co" + str(CO) + "_" + str(device)
    log_file = func_name + ".json"
    
    # print(i+1)
    # print("%dx%dx%dx%d" % (N, C, H, W))
    # print("ksize=%d, stride=%d, pad=%d" % (ksize, stride, padding))
    # print(log_file)
    # print()

    task = tvm.auto_scheduler.create_task(dense_bwd, (N, CI, CO, "float32"), target)

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
