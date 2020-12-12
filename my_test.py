import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

import torch

# no bias
# 
@auto_scheduler.register_workload
def sepconv(N, H, W, CO, CI, KH, KW, dtype):
    stride = 1
    groups = CI
    data = te.placeholder((N, CI, H, W), name="data", dtype=dtype)
    depth_kernel = te.placeholder((CI, CI // groups, KH, KW), name="depth_kernel", dtype=dtype)
    depth_conv = topi.nn.group_conv2d_nchw(data, depth_kernel, stride=stride, padding=0, dilation=1, groups=groups, out_dtype="float32")
    point_kernel = te.placeholder((CO, CI, 1, 1), name="point_kernel", dtype=dtype)
    out = topi.nn.conv2d_nchw(depth_conv, point_kernel, stride=1, padding=0, dilation=1, out_dtype="float32")
    return [data, depth_kernel, point_kernel, out]


# N, CI, H, W = (32, 32, 112, 112)
# CO, KH, KW = (64, 3, 3) 
# N, CI, H, W = (64, 3, 229, 229)
# CO, KH, KW = (10, 3, 3) 
N, CI, H, W = (3, 3, 20, 20)
CO, KH, KW = (10, 3, 3) 
target = tvm.target.Target("cuda")
log_file = "sepconv_test.json"
num_test_trails = 100
num_search_trails = 100

task = tvm.auto_scheduler.create_task(sepconv, (N, H, W, CO, CI, KH, KW, "float32"), target)
print(task.compute_dag)

###
### do search
###
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=num_search_trails,
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
del measure_ctx

# tvm result
ctx = tvm.gpu()
func = tvm.build(sch, args, target)


