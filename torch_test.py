import numpy as np
import time
import torch


num_trails = 100

N, C, H, W = 128, 30, 229, 229
ksize = (3, 3)
Cout = 72

def prelu():
    input = torch.rand((N, C, H, W), dtype=torch.float).cuda()
    prelu = torch.nn.PReLU(C).cuda()
    output = prelu(input)
    torch.cuda.synchronize()
    time_start=time.time()
    for i in range(num_trails):
        output = prelu(input)
        torch.cuda.synchronize()
    
    time_end=time.time()
    print('avg time',(time_end-time_start) * 1000 / num_trails)

def sep_conv():
    input = torch.rand((N, C, H, W), dtype=torch.float).cuda()
    depth_conv = torch.nn.Conv2d(C, C, ksize, 1, 0, 1, C, bias=True).cuda()
    point_conv = torch.nn.Conv2d(C, Cout, 1, 1, 0, 1, 1, bias=True).cuda()
    out = depth_conv(input)
    out = point_conv(out)
    torch.cuda.synchronize()
    time_start=time.time()
    for i in range(num_trails):
        out = depth_conv(input)
        out = point_conv(out)
    torch.cuda.synchronize()
    time_end=time.time()
    print('avg time',(time_end-time_start) * 1000 / num_trails)

sep_conv()