import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

@auto_scheduler.register_workload
def dense_fwd(N, CI, CO, dtype):
    data = te.placeholder((N, CI), name="data", dtype=dtype)
    weight = te.placeholder((CO, CI), name="weight", dtype=dtype)
    bias = te.placeholder((CO, ), name="bias", dtype=dtype)
    out = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, out]


device = "cuda"
target = tvm.target.Target(device)
num_search_trails = 5

time_begin = time.time()
N, CI, CO = 32, 1024, 10
func_name = "dense_fwd_n" + str(N) + "_ci" + str(CI) + "_co" + str(CO) + "_" + str(device)
log_file = func_name + ".json"

task = tvm.auto_scheduler.create_task(dense_fwd, (N, CI, CO, "float32"), target)

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
