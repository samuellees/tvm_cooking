import tvm
import numpy
from tvm import te
import timeit

matrix_size = 1024
M = matrix_size
N = matrix_size
K = matrix_size
block_size = 64

dtype = "float32"
target = 'llvm'

ctx = tvm.context(target, 0)
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), ctx)
np_repeat = 100
np_runing_time = timeit.timeit(setup='import numpy\n'
                                     'M = ' + str(M) + '\n'
                                     'K = ' + str(K) + '\n'
                                     'N = ' + str(N) + '\n'
                                     'dtype = "float32"\n'
                                     'a = numpy.random.rand(M, K).astype(dtype)\n'
                                     'b = numpy.random.rand(K, N).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=np_repeat)
print("Numpy running time: %f" % (np_runing_time / np_repeat))
answer = numpy.dot(a.asnumpy(), b.asnumpy())



A = te.placeholder((M, K), dtype='float32', name='A')
B = te.placeholder((K, N), dtype='float32', name='B')
k = te.reduce_axis((0, K), name='k')
C = te.compute((M, N), lambda i, j: te.sum(A[i, k]*B[k,j], axis=k), name='C')

# Default schedule
# s = te.create_schedule(C.op)
# func = tvm.build(s, [A, B, C], target=target, name='mmult')
# c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
# func(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
# evaluator = func.time_evaluator(func.entry_name, ctx, number=np_repeat)
# print('Baseline: %f' % evaluator(a, b, c).mean)


# s = te.create_schedule(C.op)
# k = s[C].op.reduce_axis[0]
# y, x = s[C].op.axis
# s[C].reorder(y, k, x)
# func = tvm.build(s, [A, B, C], target=target, name='mmult')
# c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
# func(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
# evaluator = func.time_evaluator(func.entry_name, ctx, number=np_repeat)
# print('reorder: %f' % evaluator(a, b, c).mean)


# s = te.create_schedule(C.op)
# k = s[C].op.reduce_axis[0]
# y, x = s[C].op.axis
# ko, ki = s[C].split(k, factor=block_size)
# yo, yi = s[C].split(y, factor=block_size)
# xo, xi = s[C].split(x, factor=block_size)
# s[C].reorder(yo, xo, ko, ki, yi, xi)
# func = tvm.build(s, [A, B, C], target=target, name='mmult')
# c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
# func(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
# evaluator = func.time_evaluator(func.entry_name, ctx, number=np_repeat)
# print('tiling1: %f' % evaluator(a, b, c).mean)


# s = te.create_schedule(C.op)
# k = s[C].op.reduce_axis[0]
# y, x = s[C].op.axis
# ko, ki = s[C].split(k, factor=block_size)
# yo, yi = s[C].split(y, factor=block_size)
# xo, xi = s[C].split(x, factor=block_size)
# s[C].reorder(yo, xo, ko, yi, ki, xi)
# func = tvm.build(s, [A, B, C], target=target, name='mmult')
# c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
# func(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
# evaluator = func.time_evaluator(func.entry_name, ctx, number=np_repeat)
# print('tiling2: %f' % evaluator(a, b, c).mean)


s = te.create_schedule(C.op)
k = s[C].op.reduce_axis[0]
y, x = s[C].op.axis
ko, ki = s[C].split(k, factor=block_size)
yo, yi = s[C].split(y, factor=block_size)
xo, xi = s[C].split(x, factor=block_size)
s[C].reorder(yo, xo, ko, yi, ki, xi)
s[C].vectorize(xi)
func = tvm.build(s, [A, B, C], target=target, name='mmult')
c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
evaluator = func.time_evaluator(func.entry_name, ctx, number=np_repeat)
print('tiling2+vec: %f' % evaluator(a, b, c).mean)


s = te.create_schedule(C.op)
k = s[C].op.reduce_axis[0]
y, x = s[C].op.axis
ko, ki = s[C].split(k, factor=block_size)
yo, yi = s[C].split(y, factor=block_size)
xo, xi = s[C].split(x, factor=block_size)
s[C].reorder(yo, xo, ko, yi, ki, xi)
s[C].vectorize(xi)
s[C].unroll(ki)
s[C].parallel(xo)
func = tvm.build(s, [A, B, C], target=target, name='mmult')
c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
evaluator = func.time_evaluator(func.entry_name, ctx, number=np_repeat)
print('tiling2+vec+parallel+unroll: %f' % evaluator(a, b, c).mean)


s = te.create_schedule(C.op)
CC = s.cache_write(C, 'global')
y, x = s[C].op.axis
yo, yi = s[C].split(y, factor=block_size)
xo, xi = s[C].split(x, factor=block_size)
s[C].reorder(yo, xo, yi, xi)

s[CC].compute_at(s[C], xo)
yc, xc = s[CC].op.axis

k, = s[CC].op.reduce_axis
ko, ki = s[CC].split(k, factor=block_size)
s[CC].reorder(ko, yc, ki, xc)

s[CC].vectorize(xc)
s[CC].unroll(ki)

s[C].parallel(xo)
func = tvm.build(s, [A, B, C], target=target, name='mmult')
c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
evaluator = func.time_evaluator(func.entry_name, ctx, number=np_repeat)
print('tiling2+vec+parallel+unroll+cache_write: %f' % evaluator(a, b, c).mean)


