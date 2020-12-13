import os
import time

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

# no bias
# 
@auto_scheduler.register_workload
def sepconv(N, H, W, CO, CI, KH, KW, stride, dtype):
    groups = CI
    data = te.placeholder((N, CI, H, W), name="data", dtype=dtype)
    depth_kernel = te.placeholder((CI, CI // groups, KH, KW), name="depth_kernel", dtype=dtype)
    depth_conv = topi.nn.group_conv2d_nchw(data, depth_kernel, stride, padding=1, dilation=1, groups=groups, out_dtype="float32")
    point_kernel = te.placeholder((CO, CI, 1, 1), name="point_kernel", dtype=dtype)
    out = topi.nn.conv2d_nchw(depth_conv, point_kernel, stride=1, padding=0, dilation=1, out_dtype="float32")
    return [data, depth_kernel, point_kernel, out]

nchw_list = [[32, 32, 112, 112],
             [32, 64, 112, 112],
             [32, 128, 56, 56],
             [32, 128, 56, 56],
             [32, 256, 28, 28],
             [32, 256, 28, 28],
             [32, 512, 14, 14],
             [32, 512, 14, 14],
             [32, 1024, 7, 7],
             [32, 1024, 7, 7]]
co_list = [64, 
           128,
           128,
           256,
           256,
           512,
           512,
           1024,
           1024]
stride_list = [1, 2, 1, 2, 1, 2, 1, 2, 1]

for i in range(len(co_list)):
    i = 0
    time_begin = time.time()
    nchw = nchw_list[i]
    stride = stride_list[i]
    N, CI, H, W = nchw
    CO, KH, KW = (co_list[i], 3, 3) 
    target = tvm.target.Target("cuda")
    func_name = "sepconv_" + str(nchw[1]) + "_" + str(co_list[i]) + "_1_" + str(stride)
    log_file = func_name + ".json"
    num_test_trails = 100
    num_search_trails = 800

    task = tvm.auto_scheduler.create_task(sepconv, (N, H, W, CO, CI, KH, KW, stride, "float32"), target)

    ###
    ### do search
    ###
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_search_trails,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    # sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
    sch, args = auto_scheduler.auto_schedule(task, search_policy, tuning_options=tune_option)
    del measure_ctx

    ###
    ### use history
    ###
    # inp, res = auto_scheduler.load_best(log_file, task.workload_key)
    # sch, args = task.compute_dag.apply_steps_from_state(inp.state)

    


    # tvm result
    ctx = tvm.gpu()
    func = tvm.build(sch, args, target, name=func_name)
    
    obj_fname = func_name + ".o"
    ptx_fname = func_name + ".ptx"
    func.save(obj_fname)
    func.imported_modules[0].save(ptx_fname)
    print("IterTime: ", (time.time() - time_begin))
    exit()
