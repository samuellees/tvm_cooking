import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

@auto_scheduler.register_workload
def gemm(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((M, N), lambda i, j: te.sum(A[i,k]*B[k,j], axis=k), name='C')
    return [A, B, C]


device = "cuda"
target = tvm.target.Target(device)
num_test_trails = 100
num_search_trails = 1000

time_begin = time.time()
M, N, K = 8196, 8196, 8196
func_name = "gemm_m" + str(N) + "_n" + str(N) + "_k" + str(K) + "_" + str(device)
log_file = func_name + ".json"

# print(i+1)
# print("%dx%dx%dx%d" % (N, C, H, W))
# print("ksize=%d, stride=%d, pad=%d" % (ksize, stride, padding))
# print(log_file)
# print()

task = tvm.auto_scheduler.create_task(gemm, (M, N, K, "float32"), target)

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
