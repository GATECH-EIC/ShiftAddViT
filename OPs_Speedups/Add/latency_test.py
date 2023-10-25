import os

import numpy as np

import tvm
from tvm import te
import tvm.testing

import time
from tvm import auto_scheduler
import torch

target = "cuda"  
dev = tvm.device(target, 0)

shapeList = [[1, 1, 3136, 32, 16],   # input:[32, 1, 3136, 32] , mask:[32, 1, 3136, 16]   
            [1, 2, 784, 32, 16],      # input:[32, 2, 784, 32] , mask:[32, 2, 784, 16]
            [1, 5, 196, 32, 16]]     # input:[32, 5, 196, 32] , mask:[32, 5, 196, 16]

@auto_scheduler.register_workload 
def matmask(BATCH, NUM, SQUEEZE, INdim, MASKdim):
    INPUT = te.placeholder((BATCH, NUM, SQUEEZE,INdim), name="INPUT", dtype="float32")
    MASK = te.placeholder((BATCH, NUM, SQUEEZE,MASKdim), name="MASK", dtype="float32")
    k = te.reduce_axis((0, SQUEEZE), name="k")
    matmask = te.compute(
        (BATCH, NUM,MASKdim,INdim),
        lambda a, i, m, n : te.sum(expr=te.if_then_else(MASK[a, i, k, m]>0, INPUT[a, i, k, n], 0),
                                axis=k),
        name="matmask",
        attrs={"layout_free_placeholders": [MASK]}, 
    )
    return [INPUT, MASK, matmask]


@auto_scheduler.register_workload  
def matmul(BATCH, NUM, SQUEEZE, INdim, MASKdim):
    INPUT = te.placeholder((BATCH, NUM,SQUEEZE,INdim), name="INPUT", dtype="float32")
    MASK = te.placeholder((BATCH, NUM,SQUEEZE,MASKdim), name="MASK", dtype="float32")
    k = te.reduce_axis((0, SQUEEZE), name="k")
    matmul = te.compute(
        (BATCH, NUM,MASKdim,INdim),
        lambda a, i, m, n : te.sum(MASK[a, i, k, m] * INPUT[a, i, k, n] ,
                                axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [MASK]}, 
    )
    return [INPUT, MASK, matmul]




for eachSize in shapeList:
    BATCH = eachSize[0]
    NUM = eachSize[1]
    SQUEEZE = eachSize[2]
    IN = eachSize[3]
    MASK = eachSize[4]

    task = tvm.auto_scheduler.SearchTask(func=matmask, args=(BATCH, NUM, SQUEEZE, IN, MASK), target=target) 
    log_file = "./result/matmask.json"
    sch, args = task.apply_best(log_file)
    matmask_cal = tvm.build(sch, args, target = target)

    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(BATCH, NUM, SQUEEZE, IN, MASK), target=target) 
    log_file = "./result/matmul.json"
    sch, args = task.apply_best(log_file)
    matmul_cal = tvm.build(sch, args, target = target)


    input_np = np.random.uniform(size=(BATCH , NUM, SQUEEZE, IN)).astype(np.float32)
    mask_np = np.random.uniform(size=(BATCH , NUM, SQUEEZE, MASK))
    weight_np = np.random.uniform(size=(BATCH , NUM, SQUEEZE, MASK)).astype(np.float32)
    mask_np = np.float32(mask_np>0)

    input_torch = torch.tensor(input_np,device="cuda",dtype=torch.float32)
    mask_torch = torch.tensor(mask_np,device="cuda",dtype=torch.float32)
    weight_torch = torch.tensor(weight_np,device="cuda",dtype=torch.float32)
    out_torch = torch.einsum("abnd,abne->abde", mask_torch * weight_torch, input_torch)

    input_tvm = tvm.nd.array(input_np, device=dev)
    mask_tvm = tvm.nd.array(mask_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(out_torch.shape, device=dev, dtype='float32')

    # warm up
    REPEAT = 10000
    time1 = time.time()
    for i in range(REPEAT):
        out_torch = torch.einsum("abnd,abne->abde", mask_torch * weight_torch, input_torch)
    torch.cuda.synchronize()
    time2 = time.time()
    print("===== calculate time test start =======")
    print(f"=> test shape: input - [{BATCH},{NUM},{SQUEEZE},{IN}], mask - [{BATCH},{NUM},{SQUEEZE},{MASK}]")


    total_time = 0
    for i in range(REPEAT):
        time1 = time.time()
        out_torch = torch.einsum("abnd,abne->abde", mask_torch, input_torch)
        torch.cuda.synchronize()
        time2 = time.time()
        total_time += time2 - time1
    print(f"pytorch einsum calculate time test (repeat:{REPEAT}): {(total_time)/10} ms")

    total_time = 0
    for i in range(REPEAT):
        time1 = time.time()
        out_torch = mask_torch.transpose(-2, -1) @ input_torch
        torch.cuda.synchronize()
        time2 = time.time()
        total_time += time2 - time1
    print(f"pytorch matmul calculate time test (repeat:{REPEAT}): {(total_time)/10} ms")


    evaluator = matmul_cal.time_evaluator(matmul_cal.entry_name, dev, repeat=3, min_repeat_ms=500)
    total_time = evaluator(input_tvm, mask_tvm, out_tvm).mean
    print(f"tvm matmul calculate time test (repeat:{REPEAT}): {total_time*1000} ms")

    evaluator = matmask_cal.time_evaluator(matmask_cal.entry_name, dev, repeat=3, min_repeat_ms=500)
    total_time = evaluator(input_tvm, mask_tvm, out_tvm).mean
    print(f"tvm matmask calculate time test (repeat:{REPEAT}): {total_time*1000} ms")


    print("===== calculate time test end =======")