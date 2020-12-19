import tvm
from tvm import te
import numpy as np

dtype = "float32"
target = 'cuda'
np_repeat = 10

matrix_size = 8192
block_size = 64
wptx = 4  # works per thread at x direction
wpty = 8
ntx = 16  # num of threads at x direction
nty = 8

M = matrix_size
N = matrix_size
K = matrix_size

# create tensor
A = te.placeholder((M, K), name='A', dtype=dtype)
B = te.placeholder((K, N), name='B', dtype=dtype)
k = te.reduce_axis((0, K), name='k')
C = te.compute((M, N), lambda i, j: te.sum(A[i,k]*B[k,j], axis=k), name='C')

# create schedule
s = te.create_schedule(C.op)

# set memory hierarchy
sharedA = s.cache_read(A, 'shared', [C])
sharedB = s.cache_read(B, 'shared', [C])
regA = s.cache_read(sharedA, 'local', [C])
regB = s.cache_read(sharedB, 'local', [C])
regC = s.cache_write(C, 'local')

# get thread and block idx
block_y = te.thread_axis('blockIdx.y')
block_x = te.thread_axis('blockIdx.x')
thread_yi = te.thread_axis((0, nty), 'threadIdx.y')
thread_xi = te.thread_axis((0, ntx), 'threadIdx.x')
thread_yo = te.thread_axis((0, wpty), "vthread", name="vy")
thread_xo = te.thread_axis((0, wptx), "vthread", name="vx")

# describe how to split workloads(C matrix)
y, x = s[C].op.axis
by, y_ = s[C].split(y, factor=block_size)
bx, x_ = s[C].split(x, factor=block_size)
yo, yi = s[C].split(y_, nparts=wpty) # virtual split
xo, xi = s[C].split(x_, nparts=wptx) # virtual split
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
ko, ki = s[regC].split(k, factor=block_size)
s[regC].reorder(ko, ki)
s[regC].unroll(ki)

# describe how to load sharedA and regA
s[sharedA].compute_at(s[regC], ko)
s[regA].compute_at(s[regC], ki)
yA, xA = s[sharedA].op.axis
byA, yA_ = s[sharedA].split(yA, factor=block_size)
bxA, xA_ = s[sharedA].split(xA, factor=block_size)
yoA, yiA = s[sharedA].split(yA_, nparts=wpty) # virtual split
xoA, xiA = s[sharedA].split(xA_, nparts=wptx) # virtual split
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
byB, yB_ = s[sharedB].split(yB, factor=block_size)
bxB, xB_ = s[sharedB].split(xB, factor=block_size)
yoB, yiB = s[sharedB].split(yB_, nparts=wpty) # virtual split
xoB, xiB = s[sharedB].split(xB_, nparts=wptx) # virtual split
s[sharedB].bind(yiB, thread_yi)
s[sharedB].bind(xiB, thread_xi)
s[sharedB].reorder(byB, bxB, yiB, xiB, yoB, xoB)
s[sharedB].vectorize(xoB)
s[sharedB].unroll(xoB)
s[sharedB].unroll(yoB)

# answer from numpy
ctx = tvm.context(target, 0)
a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
answer = np.dot(a.asnumpy(), b.asnumpy())

# answer from tvm
func = tvm.build(s, [A, B, C], target=target, name='gemm')
c = tvm.nd.array(np.zeros((M, N), dtype=dtype), ctx)
func(a, b, c)
np.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
evaluator = func.time_evaluator(func.entry_name, ctx, number=np_repeat)
evaluator(a, b, c)
time = evaluator(a, b, c).mean
flops = 2.0*M/1000*N/1000*K/1000/time
print('tvm time: %f s' % time)
print("flops: %f" % flops)

# flops of CUDA optimize: 6.378 T
# flops of tvm optimize: 6.592 T


# print(tvm.lower(s, [A, B, C], simple_mode=True))

# CUDA source
# func = tvm.build(s, [A, B, C], 'cuda')
# dev_module = func.imported_modules[0]
# print(dev_module.get_source())