import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.nn.utils import get_pad_tuple


@auto_scheduler.register_workload
def batch_norm_fwd(N, C, H, W, dtype="float32"):
  dshape = (N, C, H, W)
  oshape = (C, )
  bshape = (1, C, 1, 1)
  sshape = (1, )
  data = te.placeholder(dshape, name="data", dtype=dtype)
  scale = te.placeholder(oshape, name="scale", dtype=dtype)
  bias = te.placeholder(oshape, name="bias", dtype=dtype)
  running_mean = te.placeholder(oshape, name="running_mean", dtype=dtype)
  running_var = te.placeholder(oshape, name="running_var", dtype=dtype)
  eps = te.placeholder(sshape, name="eps", dtype=dtype)
  momentum = te.placeholder(sshape, name="momentum", dtype=dtype)

  axis = (0, 2, 3)
  num_ele = dshape[0] * dshape[2] * dshape[3]
  frac_num_ele = 1.0 / num_ele
  # compute batch mean
  mean_sum = topi.sum(data, axis, keepdims=True)
  saved_mean = topi.multiply(mean_sum, frac_num_ele)
  # compute batch rvars
  var_sub = topi.subtract(data, saved_mean)
  var_mul = topi.multiply(var_sub, var_sub)
  var_sum = topi.sum(var_mul, axis, keepdims=True)
  var = topi.multiply(var_sum, frac_num_ele)
  output_add = topi.add(var, eps)
  saved_rvars = topi.sqrt(output_add)
  # # compute output
  output_sub = topi.subtract(data, saved_mean)
  output_norm = topi.divide(output_sub, saved_rvars)
  scale_board = topi.reshape(scale, bshape)
  bias_board = topi.reshape(bias, bshape)
  output = topi.add(topi.multiply(output_norm, scale_board), bias_board)
  # reshape saved_rvars
  saved_rvars = topi.reshape(saved_rvars, oshape)
  # update running mean
  running_mean_mul1 = topi.multiply(running_mean, topi.subtract(1.0, momentum))
  running_mean_mul2 = topi.multiply(topi.reshape(saved_mean, oshape), momentum)
  running_mean_out = topi.add(running_mean_mul1, running_mean_mul2)
  # update running var
  saved_var_mul1 = topi.multiply(running_var, topi.subtract(1.0, momentum))
  saved_var_mul2 = topi.multiply(topi.reshape(var, oshape), momentum)
  running_var_out = topi.add(saved_var_mul1, saved_var_mul2)
  # reshape saved_mean
  saved_mean = topi.reshape(saved_mean, oshape)

  return [data, scale, bias, running_mean, running_var, momentum, eps, 
            output, saved_mean, saved_rvars, running_mean_out, running_var_out]


num_search_trails = 2
N, C, H, W = 10, 10, 10, 10

time_begin = time.time()
target = tvm.target.Target("cuda")
func_name = "batch_norm_fwd_n" + str(N) + "_c" + str(C) + "_h" + str(H) + "_w" + str(W) + "_" + "cuda"
log_file = func_name + ".json"

task = tvm.auto_scheduler.create_task(batch_norm_fwd, (N, C, H, W, "float32"), target)

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

