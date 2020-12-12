import os

import numpy as np
import tvm
from tvm import te, auto_scheduler

import torch


@auto_scheduler.register_workload
def prelu(N, C, H, W, dtype):
    X = te.placeholder((N, C, H, W), name="X", dtype=dtype) # input
    A = te.placeholder((C,), name="A", dtype=dtype)          # alphas

    out = te.compute((N, C, H, W), lambda n, c, h, w: 
            tvm.tir.if_then_else(X[n, c, h, w] > 0, X[n, c, h, w], X[n, c, h, w] * A[c]), name="out")

    return [X, A, out]


N, C, H, W = (4, 2, 2, 2)
target = tvm.target.Target("cuda")
log_file = "prelu.json"
num_trails = 100

task = tvm.auto_scheduler.create_task(prelu, (N, C, H, W, "float32"), target)
# print(task.compute_dag)

###
### do search
###
# measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
# tune_option = auto_scheduler.TuningOptions(
#     num_measure_trials=10,
#     runner=measure_ctx.runner,
#     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
#     verbose=2,
# )
# sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
# del measure_ctx

###
### use history
###
inp, res = auto_scheduler.load_best(log_file, task.workload_key)
sch, args = task.compute_dag.apply_steps_from_state(inp.state)


# print(tvm.lower(sch, args, simple_mode=True))

func = tvm.build(sch, args, target, name="prelu")


func.save("a.o")
func.imported_modules[0].save('a.ptx')

ttt = tvm.runtime.load_module("a.o")
ttt_dev = tvm.runtime.load_module("a.ptx")
ttt.import_module(ttt_dev)
func = ttt.get_function("prelu")


x_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
a_np = np.random.uniform(size=(C, 1, 1)).astype(np.float32)
x_np_less0_idx = (x_np < 0)
x_np_less0 = np.where(x_np_less0_idx, x_np, 0)
x_np_great0 = x_np - x_np_less0
y_np = x_np_less0 * a_np + x_np_great0

ctx = tvm.gpu()
x_tvm = tvm.nd.array(x_np, ctx=ctx) 
a_tvm = tvm.nd.array(a_np.reshape((C,)), ctx=ctx)
y_tvm = tvm.nd.empty(y_np.shape, ctx=ctx)
func(x_tvm, a_tvm, y_tvm)

print(','.join([str(e) for e in x_np.reshape(32)]))
print(a_np)
print(y_np.reshape(32))

# Check results
np.testing.assert_allclose(y_np, y_tvm.asnumpy(), rtol=1e-3)

# # Evaluate execution time.
# evaluator = func.time_evaluator(func.entry_name, ctx, repeat=num_trails, min_repeat_ms=500)
# print("Execution time of tvm operator: %.3f ms"
#     % (np.average(evaluator(x_tvm, a_tvm, y_tvm).results) * 1000)
# )


