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
    C = te.compute((M, N), 
        lambda i, j: te.sum(A[i,k]*B[k,j], axis=k), name='C')
    return [A, B, C]


device = "llvm -mcpu=core-avx2"
target = tvm.target.Target(device)
num_test_trails = 50
num_search_trails = 1000
dtype = "float32"

matrix_size = 4096
M, K, N = matrix_size, matrix_size, matrix_size
func_name = "gemm_m" + str(N) + "_n" + str(N) + "_k" + str(K) + "_" + str("llvm-mcpu=core-avx2")
log_file = func_name + ".json"

task = tvm.auto_scheduler.create_task(gemm, (M, N, K, dtype), target)
# cost_model = auto_scheduler.XGBModel()
# cost_model.update_from_file(log_file)
# search_policy = auto_scheduler.SketchPolicy(
#     task,
#     cost_model,
#     init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)],
# )
### search
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=num_search_trails,
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
# sch, args = auto_scheduler.auto_schedule(task, search_policy, tuning_options=tune_option)
sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
del measure_ctx

### load history
# inp, res = auto_scheduler.load_best(log_file, task.workload_key)
# sch, args = task.compute_dag.apply_steps_from_state(inp.state)

# build func
ctx = tvm.cpu()
func = tvm.build(sch, args, target, name=func_name)
# save result
# obj_fname = func_name + ".o"
# ptx_fname = func_name + ".ptx"
# func.save(obj_fname)
# func.imported_modules[0].save(ptx_fname)

a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
c = tvm.nd.array(np.zeros((M, N), dtype=dtype), ctx)

evaluator = func.time_evaluator(func.entry_name, ctx, number=num_test_trails)
print('Time: %f' % evaluator(a, b, c).mean)

