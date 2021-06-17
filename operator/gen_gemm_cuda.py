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
# M, K, N = 4096, 4096, 4096
M, K, N = 8000, 8000, 8000
func_name = "gemm_m" + str(N) + "_n" + str(N) + "_k" + str(K) + "_" + str(device)
log_file = func_name + ".json"

# print(i+1)
# print("%dx%dx%dx%d" % (N, C, H, W))
# print("ksize=%d, stride=%d, pad=%d" % (ksize, stride, padding))
# print(log_file)
# print()

task = tvm.auto_scheduler.create_task(gemm, (M, N, K, "float32"), target)

### search
# measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
# tune_option = auto_scheduler.TuningOptions(
#     num_measure_trials=num_search_trails,
#     runner=measure_ctx.runner,
#     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
#     verbose=2,
# )
# sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
# del measure_ctx

# ### load history
# inp, res = auto_scheduler.load_best(log_file, task.workload_key)
# sch, args = task.compute_dag.apply_steps_from_state(inp.state)

# # build func
# ctx = tvm.gpu()
# func = tvm.build(sch, args, target, name=func_name)
# # save result
# obj_fname = func_name + ".o"
# ptx_fname = func_name + ".ptx"
# func.save(obj_fname)
# func.imported_modules[0].save(ptx_fname)

# time_end = time.time()
# print("IterTime: ", (time_end - time_begin))



func_module = tvm.runtime.load_module(func_name + ".o")
func_module_dev = tvm.runtime.load_module(func_name + ".ptx")
func_module.import_module(func_module_dev)
func = func_module.get_function(func_name)

A = np.random.uniform(size=(M, K)).astype(np.float32)
B = np.random.uniform(size=(K, N)).astype(np.float32)

# tvm result
ctx = tvm.gpu()
A_tvm = tvm.nd.array(A, ctx=ctx) 
B_tvm = tvm.nd.array(B, ctx=ctx)
C_tvm = tvm.nd.empty((M, N), ctx=ctx)
# func(A_tvm, B_tvm, C_tvm)

# tvm timing.
evaluator = func_module.time_evaluator(func_module.entry_name, ctx, repeat=num_test_trails, min_repeat_ms=500)
evaluator(A_tvm, B_tvm, C_tvm)
time = evaluator(A_tvm, B_tvm, C_tvm).mean
flops = 2.0*M/1000*N/1000*K/1000/time
print('tvm time: %f s' % time)
print("flops: %f" % flops)

# with cuda-11 tvm-0.8
# 6.614 Flops