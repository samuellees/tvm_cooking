import tvm
from tvm import te
from tvm import autotvm
import numpy
import logging
import sys

@autotvm.template("examples/gemm_tune")
def gemm_tune(M, N, K, dtype):

  # create tensor
  A = te.placeholder((M, K), name='A', dtype=dtype)
  B = te.placeholder((K, N), name='B', dtype=dtype)
  k = te.reduce_axis((0, K), name='k')
  C = te.compute((M, N), lambda i, j: te.sum(A[i,k]*B[k,j], axis=k), name='C')

  # create schedule
  s = te.create_schedule(C.op)
  y, x = s[C].op.axis

  # set memory hierarchy
  sharedA = s.cache_read(A, 'shared', [C])
  sharedB = s.cache_read(B, 'shared', [C])
  regA = s.cache_read(sharedA, 'local', [C])
  regB = s.cache_read(sharedB, 'local', [C])
  regC = s.cache_write(C, 'local')

  # define search space
  cfg = autotvm.get_config()
  cfg.define_split("tile_y", y, num_outputs=3)
  cfg.define_split("tile_x", x, num_outputs=3)
  by, yo, yi = cfg["tile_y"].apply(s, C, y)
  bx, xo, xi = cfg["tile_x"].apply(s, C, x)

  # get thread and block idx
  block_y = te.thread_axis('blockIdx.y')
  block_x = te.thread_axis('blockIdx.x')
  thread_yi = te.thread_axis('threadIdx.y')
  thread_xi = te.thread_axis('threadIdx.x')
  thread_yo = te.thread_axis("vthread", name="vy")
  thread_xo = te.thread_axis("vthread", name="vx")

  # describe how to split workloads(C matrix)
  s[C].bind(by, block_y)
  s[C].bind(bx, block_x)
  s[C].bind(yo, thread_yo)
  s[C].bind(xo, thread_xo)
  s[C].bind(yi, thread_yi)
  s[C].bind(xi, thread_xi)
  s[C].reorder(by, bx, yi, xi, yo, xo)

  # describe how to load regC
  s[regC].compute_at(s[C], xo)
  k, = s[regC].op.reduce_axis
  cfg.define_split("split_k", k, num_outputs=2)
  ko, ki = cfg["tile_y"].apply(s, regC, k)
  s[regC].reorder(ko, ki)
  s[regC].unroll(ki)

  # describe how to load sharedA and regA
  s[sharedA].compute_at(s[regC], ko)
  s[regA].compute_at(s[regC], ki)
  yA, xA = s[sharedA].op.axis
  cfg.define_split("split_yA", yA, num_outputs=3)
  cfg.define_split("split_xA", xA, num_outputs=3)
  byA, yoA, yiA = cfg["split_yA"].apply(s, sharedA, yA)
  bxA, xoA, xiA = cfg["split_xA"].apply(s, sharedA, xA)
  s[sharedA].bind(yiA, thread_yi)
  s[sharedA].bind(xiA, thread_xi)
  s[sharedA].reorder(byA, bxA, yiA, xiA, yoA, xoA)
  s[sharedA].vectorize(xoA)
  s[sharedA].unroll(xoA)
  s[sharedA].unroll(yoA)

  # describe how to load sharedB and regB
  s[sharedB].compute_at(s[regC], ko)
  s[regB].compute_at(s[regC], ki)
  yB, xB = s[sharedB].op.axis
  cfg.define_split("split_yB", yB, num_outputs=3)
  cfg.define_split("split_xB", xB, num_outputs=3)
  byB, yoB, yiB = cfg["split_yB"].apply(s, sharedB, yB)
  bxB, xoB, xiB = cfg["split_xB"].apply(s, sharedB, xB)
  s[sharedB].bind(yiB, thread_yi)
  s[sharedB].bind(xiB, thread_xi)
  s[sharedB].reorder(byB, bxB, yiB, xiB, yoB, xoB)
  s[sharedB].vectorize(xoB)
  s[sharedB].unroll(xoB)
  s[sharedB].unroll(yoB)

  return s, [A, B, C]




dtype = "float32"
target = 'cuda'

matrix_size = 1024
n_trail = 10
M = matrix_size
N = matrix_size
K = matrix_size

# answer from numpy
ctx = tvm.context(target, 0)
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), ctx)
answer = numpy.dot(a.asnumpy(), b.asnumpy())

# get auto tvm config space
task = autotvm.task.create("examples/gemm_tune", args=(M, N, K, dtype), target=target)
print(task.config_space)
# logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
# measure options
measure_option = autotvm.measure_option(builder='local', runner=autotvm.LocalRunner(number=5))
# set random tunner
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=n_trail,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file("gemm.log")])
# apply best from log file
with autotvm.apply_history_best("gemm.log"):
  with tvm.target.create(target):
    s, arg_bufs = gemm_tune(M, N, K, dtype)
    func = tvm.build(s, arg_bufs)
# check correntness
c_tvm = tvm.nd.array(numpy.random.rand(M, N).astype(dtype), ctx)
func(a, b, c_tvm)
tvm.testing.assert_allclose(answer, c_tvm.asnumpy(), rtol=1e-2)
print("tune success!")


# time = evaluator(a, b, c).mean
# flops = 2.0*M/1000*N/1000*K/1000/time

# print('tvm: %f' % time)
# print("flops: %f" % flops)

# flops of manual optimize: 6.378 T
# flops of tvm optimize: 6.592 T

# print(tvm.lower(s, [A, B, C], simple_mode=True))

