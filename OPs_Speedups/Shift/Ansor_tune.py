
import os

from tvm import tir
from tvm import te, auto_scheduler


shapeList = [[1, 3136, 64, 512],
            [1, 3136, 512, 64],
            [1, 784, 128, 1024],
            [1, 784, 1024, 128]]

target = "cuda"

@auto_scheduler.register_workload 
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


@auto_scheduler.register_workload 
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

@auto_scheduler.register_workload
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



os.makedirs("./result", exist_ok=True)
for eachSize in shapeList:
    A = eachSize[0]
    M = eachSize[1]
    K = eachSize[2]
    N = eachSize[3]
    print(f" ===> start opitimize shape: input - [{A},{M},{K}], weight - [{N},{K}]")

    # ================== matshift ==================

    print(" start tune matshift")
    task = auto_scheduler.SearchTask(func=matshift, args=(A, M, K, N), target=target)
    print("Computational DAG:")
    print(task.compute_dag)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=3)
    log_file = f"./result/matshift_{A}_{M}_{K}_{N}.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100000,
        early_stopping = 500,
        runner=measure_ctx.runner,
        # runner = auto_scheduler.LocalRunner(repeat=30, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)

    # ================== matshift_fake ==================

    print(" start tune matshift_fake")
    task = auto_scheduler.SearchTask(func=matshift_fake, args=(A, M, K, N), target=target)
    print("Computational DAG:")
    print(task.compute_dag)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=3)
    log_file = f"./result/matshift_fake_{A}_{M}_{K}_{N}.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100000,
        early_stopping = 500,
        runner=measure_ctx.runner,
        # runner = auto_scheduler.LocalRunner(repeat=30, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)


    # ================== matmul ==================

    print(" start tune matmul")
    task = auto_scheduler.SearchTask(func=matmul, args=(A, M, K, N), target=target)
    print("Computational DAG:")
    print(task.compute_dag)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=3)
    log_file = f"./result/matmul_{A}_{M}_{K}_{N}.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100000,
        early_stopping = 500,
        runner=measure_ctx.runner,
        # runner = auto_scheduler.LocalRunner(repeat=30, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)