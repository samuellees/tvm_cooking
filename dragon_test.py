from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np

m = 1024

# declare a matrix element-wise multiply
A = te.placeholder((m,), name="A")
B = te.placeholder((m,), name="B")
C = te.compute((m,), lambda i: A[i] * B[i], name="C")
# create schedule
s = te.create_schedule([C.op])

print(tvm.lower(s, [A, B, C], simple_mode=True))

print("----------cutting line-----------")

s[C].split(C.op.axis[0], factor=32)
print(tvm.lower(s, [A, B, C], simple_mode=True))

print("for (i: int32, 0, 1024) {") 
print("    C_2[i] = ((float32*)A_2[i]*(float32*)B_2[i]) ") 
print("} ") 
print() 
print("----------cutting line----------- ") 
print() 
print("for (i.outer: int32, 0, 32) {") 
print("    for (i.inner: int32, 0, 32) {") 
print("        C_2[((i.outer*32) + i.inner)] = ((float32*)A_2[((i.outer*32) + i.inner)]*(float32*)B_2[((i.outer*32) + i.inner)])") 
print("    }") 
print("}") 