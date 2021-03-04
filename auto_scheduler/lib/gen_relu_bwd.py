import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

@auto_scheduler.register_workload
def relu_bwd(N, C, H, W, dtype):
    shape = (N, C, H, W)
    data = te.placeholder(shape, name="data", dtype=dtype)
    grad_out = te.placeholder(shape, name="grad_out", dtype=dtype)

    grad_data = te.compute(
        (N, C, H, W), 
        lambda i, j, k, w: tvm.tir.if_then_else(data[i, j, k, w] > 0.0, grad_out[i, j, k, w] * 1.0, 0.0) * 1.0, 
        name="grad_data",
    )

    return [data, grad_out, grad_data]


device = "cuda"
target = tvm.target.Target(device)
N, C, H, W = 32, 32, 224, 224
num_search_trails = 100


time_begin = time.time()
func_name = "relu_bwd_n" + str(N) + "_c" + str(C) + "_h" + str(H) + "_w" + str(W) + "_" + str(device)
log_file = func_name + ".json"

task = tvm.auto_scheduler.create_task(relu_bwd, (N, C, H, W, "float32"), target)

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
