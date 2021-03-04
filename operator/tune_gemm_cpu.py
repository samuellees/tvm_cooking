
import logging
import sys

import numpy as np

import tvm
from tvm import te
from tvm import autotvm
# logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

@autotvm.template("examples/gemm_v1")
def gemm_v1(M, N, K, dtype):
  A = te.placeholder((M, K), name='A', dtype=dtype)
  B = te.placeholder((K, N), name='B', dtype=dtype)
  # compute
  k = te.reduce_axis((0, K), name='k')
  C = te.compute((M, N), 
    lambda i, j: te.sum(A[i, k]*B[k, j], axis=k), name='C')
  # schedule
  s = te.create_schedule(C.op)
  y, x = s[C].op.axis
  k = s[C].op.reduce_axis[0]
  # define search space
  cfg = autotvm.get_config()
  cfg.define_split("tile_x", x, num_outputs=2)
  cfg.define_split("tile_y", y, num_outputs=2)
  cfg.define_split("tile_k", k, num_outputs=2)
  # apply config
  xo, xi = cfg["tile_x"].apply(s, C, x)
  yo, yi = cfg["tile_y"].apply(s, C, y)
  ko, ki = cfg["tile_k"].apply(s, C, k)
  # define order
  # cfg.define_reorder("reorder", [yo, xo, ko, yi, ki, xi], "all") 
  # cfg["reorder"].apply(s, C, [yo, xo, ko, yi, ki, xi])

  # other
  s[C].reorder(yo, xo, ko, yi, ki, xi)
  s[C].vectorize(xi)
  s[C].unroll(ki)
  s[C].parallel(xo)
  return s, [A, B, C]


matrix_size = 4096
M = matrix_size
N = matrix_size
K = matrix_size
n_trail = 500
np_repeat = 100
log_file = './log/tune_gemm_cpu' + str(matrix_size) + ".log"

target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(target, 0)

# get auto tvm config space
task = autotvm.task.create("examples/gemm_v1", args=(M, N, K, 'float32'), target=target)
print(task.config_space)
# measure options
measure_option = autotvm.measure_option(builder='local', runner=autotvm.LocalRunner(number=5))
# set tunner
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=n_trail,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file(log_file)])
# apply best from log file
with autotvm.apply_history_best(log_file):
  with tvm.target.Target(target):
    s, arg_bufs = gemm_v1(M, N, K, 'float32')
    func = tvm.build(s, arg_bufs)
# check correntness
a_np = np.random.uniform(size=(M, K)).astype(np.float32)
b_np = np.random.uniform(size=(K, N)).astype(np.float32)
c_np = a_np.dot(b_np)
c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
np.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
# timing
evaluator = func.time_evaluator(func.entry_name, ctx, number=np_repeat)
print('auto: %f' % evaluator(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm).mean)


