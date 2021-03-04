import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.nn.utils import get_pad_tuple


@auto_scheduler.register_workload
def conv_bwd(N, CI, HI, WI, CO, HO, WO, KSIZE, stride, padding, dtype):
    strides = (stride, stride)
    shape_data = (N, CI, HI, WI)
    shape_weight = (CO, CI, KSIZE, KSIZE)
    shape_grad_output = (N, CO, HO, WO)
    # given tensor
    data = te.placeholder(shape_data, name="data", dtype=dtype)
    weight = te.placeholder(shape_weight, name="weight", dtype=dtype)
    grad_output = te.placeholder(shape_grad_output, name="grad_output", dtype=dtype)
    # grad_data
    out_h = (HO - 1) * strides[0] - 2 * padding + KSIZE
    out_w = (WO - 1) * strides[1] - 2 * padding + KSIZE
    output_padding = (HI - out_h, WI - out_w)
    grad_data = topi.nn.conv2d_transpose_nchw(grad_output, weight, strides, padding, dtype, output_padding)
    # grad_weight
    dilation_h, dilation_w = (1, 1)
    batch, in_channel, in_h, in_w = shape_data
    out_channel, _, filter_h, filter_w = shape_weight
    grad_output_tmp = topi.tile(grad_output, [1, in_channel, 1, 1])
    grad_output_tmp = topi.reshape(grad_output_tmp, [batch * in_channel * out_channel, 1, HO, WO])
    data_tmp = topi.reshape(data, [1, in_channel * batch, HI, WI])
    grad_weight = topi.nn.group_conv2d_nchw(data_tmp, grad_output_tmp, stride=(dilation_h, dilation_w), 
        padding=padding, dilation=strides, groups=in_channel * batch, out_dtype=dtype)
    # infer shape of grad_weight
    _, _, grad_h, grad_w = shape_grad_output
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    padded_weight_grad_h = (in_h - (grad_h - 1) * strides[0] - 1 + fpad_top + fpad_bottom) // dilation_h + 1
    padded_weight_grad_w = (in_w - (grad_w - 1) * strides[1] - 1 + fpad_left + fpad_right) // dilation_w + 1
    grad_weight = topi.reshape(
        grad_weight, [batch, in_channel, out_channel, padded_weight_grad_h, padded_weight_grad_w])
    grad_weight = topi.sum(grad_weight, axis=0)
    grad_weight = topi.transpose(grad_weight, [1, 0, 2, 3])

    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        grad_weight = topi.strided_slice(grad_weight, begin=[0, 0, 0, 0],
            end=[out_channel, in_channel, filter_h, filter_w])
        return [data, weight, grad_output, grad_data, grad_weight]

    return [data, weight, grad_output, grad_data, grad_weight]


num_search_trails = 2
N, CI, HI, WI = 8, 3, 10, 10
CO, HO, WO = 10, 10, 10
ksize = 3
stride = 1
padding = 1

time_begin = time.time()
target = tvm.target.Target("cuda")
func_name = "conv_bwd_n" + str(N) + "_c" + str(CI) + "_hi" + str(HI) + "_wi" + str(WI) + "_co"  \
      + str(CO) + "_ho" + str(HO) + "_wo" + str(WO)+ "_k" + str(ksize) + "_s" + str(stride) + "_p" + str(padding) + "_" + "cuda"
log_file = func_name + ".json"

task = tvm.auto_scheduler.create_task(conv_bwd, (N, CI, HI, WI, CO, HO, WO, ksize, stride, padding, "float32"), target)

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

