import numpy as np
import tvm
from tvm import te,tir
import tvm.testing

# the module is called `autotvm`
import time
from tvm import auto_scheduler
import torch

target = 'cuda'
dev = tvm.device(target,0)

shapeList = [[1, 3136, 64, 512],
            [1, 3136, 512, 64],
            [1, 784, 128, 1024],
            [1, 784, 1024, 128]]


@auto_scheduler.register_workload  # 注意 auto_scheduler 装饰器
def matshift(A, M, K, N):
    INPUT = te.placeholder((A, M, K), name="INPUT", dtype="int32")
    SHIFT = te.placeholder((N, K), name="SHIFT", dtype="int8")
    SIGN = te.placeholder((N, K), name="SIGN", dtype="int8")
    k = te.reduce_axis((0, K), name="k")
    matshift = te.compute(
        (A, M, N),
        # lambda a, i, j: te.sum(expr=tvm.tir.if_then_else((SIGN[j, k] > tir.const(0,dtype="int8")),   # if sign == 1
        #         (INPUT[a, i, k] >> SHIFT[j, k]),
        #         ~(INPUT[a, i, k] >> SHIFT[j, k]) + 1),
        #         axis=k),
        lambda a, i, j: te.sum(expr= SIGN[j, k] * (INPUT[a, i, k] >> SHIFT[j, k]),
                axis=k),
        name="matshift",
        attrs={"layout_free_placeholders": [SHIFT,SIGN]}, 
    )
    return [INPUT, SHIFT,SIGN, matshift]


@auto_scheduler.register_workload  # 注意 auto_scheduler 装饰器
def matmul(A, M, K, N):
    INPUT = te.placeholder((A, M, K), name="INPUT", dtype="float32")
    WEIGHT = te.placeholder((N, K), name="WEIGHT", dtype="float32")
    k = te.reduce_axis((0, K), name="k")
    matshift = te.compute(
        (A, M, N),
        lambda a, i, j: te.sum(expr= INPUT[a, i, k] * WEIGHT[j, k],
                axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [INPUT,WEIGHT]}, 
    )
    return [INPUT, WEIGHT, matshift]

@auto_scheduler.register_workload  # 注意 auto_scheduler 装饰器
def matshift_fake(bs, token, inDim, outDim):
    INPUT = te.placeholder((bs, token, inDim), name="INPUT", dtype="float32")
    SHIFT = te.placeholder((outDim, inDim), name="SHIFT", dtype="float32")
    SIGN = te.placeholder((outDim, inDim), name="SIGN", dtype="float32")
    i = te.reduce_axis((0, inDim), name="i")

    weight = te.compute(
        (outDim, inDim),
        lambda o, j: tir.pow(2, -SHIFT[o, j]) * SIGN[o, j],
        name="weight",
        attrs={"layout_free_placeholders": [SHIFT,SIGN]},
    )

    matshift = te.compute(
        (bs, token, outDim),
        lambda b, t, o: te.sum(expr= INPUT[b, t, i] * weight[o, i], axis=i),
        name="matshiftOut",
        attrs={"layout_free_placeholders": [INPUT, weight]}, 
    )
    return [INPUT, SHIFT, SIGN, matshift]

for eachSize in shapeList:
    A = eachSize[0]
    M = eachSize[1]
    K = eachSize[2]
    N = eachSize[3]

    # build matshift callable function
    task = auto_scheduler.SearchTask(func=matshift, args=(A , M , K , N), target=target)
    log_file = "./result/matshift.json"
    sch, args = task.apply_best(log_file)
    matshift_cal = tvm.build(sch, args, target = target)

    # build matmul callable function
    task = auto_scheduler.SearchTask(func=matmul, args=(A , M , K , N), target=target)
    log_file = "./result/matmul.json"
    sch, args = task.apply_best(log_file)
    matmul_cal = tvm.build(sch, args, target = target)

    # build matshift_fake callable function
    task = auto_scheduler.SearchTask(func=matshift_fake, args=(A , M , K , N), target=target)
    log_file = "./result/matshift_fake.json"
    sch, args = task.apply_best(log_file)
    matshift_fake_cal = tvm.build(sch, args, target = target)


    a_np = np.random.uniform(size=(A, M, K)).astype(np.int32)
    b_np = np.random.uniform(size=(N, K)).astype(np.int8)
    b_trans = b_np.transpose()
    c_np = np.sign(np.random.uniform(size=(N, K)).astype(np.int8))
    out_np = a_np.dot(b_trans).astype(np.int32)

    a_torch = torch.tensor(a_np,device="cuda",dtype=torch.float32)
    # b_torch = torch.tensor(b_np,device="cuda",dtype=torch.float32)    
    b_torch = torch.rand(b_np.shape,device="cuda",dtype=torch.float32)
    c_torch = torch.tensor(c_np,device="cuda",dtype=torch.float32)
    out_torch = torch.nn.functional.linear(a_torch,b_torch)

    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev, dtype='int32')
    matshift_cal(a_tvm, b_tvm,c_tvm, out_tvm)

    mul_a_tvm = tvm.nd.array(a_torch.to("cpu"), device=dev)
    mul_b_tvm = tvm.nd.array(b_torch.to("cpu"), device=dev)
    mul_c_tvm = tvm.nd.array(c_torch.to("cpu"), device=dev)
    mul_out_tvm = tvm.nd.empty(out_np.shape, device=dev, dtype='float32')
    matmul_cal(mul_a_tvm, mul_b_tvm,mul_c_tvm, mul_out_tvm)


    REPEAT = 10000  
    for i in range(REPEAT):
        output = torch.nn.functional.linear(a_torch,b_torch)
    torch.cuda.synchronize()
    print("===== calculate time test start =======")
    print(f"=> test shape: input - [{A},{M},{K}], weight - [{K},{N}]")

    # tvm matshift calculate time test
    evaluator = matshift_cal.time_evaluator(matshift_cal.entry_name, dev, min_repeat_ms = 2000)
    print(f"tvm matshift calculate time test (repeat:{REPEAT}): {evaluator(a_tvm, b_tvm,c_tvm, out_tvm).mean*1000} ms")

    # tvm matmul calculate time test
    evaluator = matmul_cal.time_evaluator(matmul_cal.entry_name, dev, min_repeat_ms = 2000)
    print(f"tvm matmul calculate time test (repeat:{REPEAT}): {evaluator(mul_a_tvm, mul_b_tvm,mul_c_tvm, mul_out_tvm).mean*1000} ms")

    evaluator = matshift_fake_cal.time_evaluator(matshift_fake_cal.entry_name, dev, min_repeat_ms = 2000)
    print(f"tvm matshift_fake calculate time test (repeat:{REPEAT}): {evaluator(mul_a_tvm, mul_b_tvm, mul_c_tvm, mul_out_tvm).mean*1000} ms")

    # torch matmul calculate time test
    timeAll = 0
    # weight = (torch.ones_like(b_torch,device="cuda") << b_torch) * c_torch
    for _ in range(REPEAT):
        tic1 = time.time()
        weight = (2 ** b_torch) * c_torch
        output = torch.nn.functional.linear(a_torch, weight * weight)
        torch.cuda.synchronize()
        tic2 = time.time()
        timeAll += tic2 - tic1
    print(f"pytorch matmul calculate time test (repeat:{REPEAT}): {(timeAll)/10} ms")
