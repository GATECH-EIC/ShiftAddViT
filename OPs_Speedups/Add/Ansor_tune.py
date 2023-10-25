import tvm
from tvm import te
import tvm.testing
from tvm import tir, auto_scheduler
import numpy as np

#input :v    mask: q,k 
shapeList = [[1, 1, 3136, 32, 16],   # input:[32, 1, 3136, 32] , mask:[32, 1, 3136, 16]   
            [1, 2, 784, 32, 16],      # input:[32, 2, 784, 32] , mask:[32, 2, 784, 16]
            [1, 5, 196, 32, 16]]     # input:[32, 5, 196, 32] , mask:[32, 5, 196, 16]

# shapeList = [[1, 2, 784, 64, 64]] 

target = "cuda"

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

os.makedirs("./result", exist_ok=True)
for eachSize in shapeList:

    BATCH = eachSize[0]
    NUM = eachSize[1] 
    SQUEEZE = eachSize[2]
    IN = eachSize[3]
    MASK = eachSize[4]


    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=10)

    print(f"start opitimize matmask, shape: input - [{BATCH},{NUM},{SQUEEZE},{IN}], mask - [{BATCH},{NUM},{SQUEEZE},{MASK}]")
    
    task = auto_scheduler.SearchTask(func=matmask,
                                    args=(BATCH, NUM, SQUEEZE, IN, MASK),
                                    target=target,
                                    layout_rewrite_option = auto_scheduler.LayoutRewriteOption.INSERT_TRANSFORM_STAGE)
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "./result/matmask.json"  
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100000, 
        early_stopping = 500,     
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2, 
    )
    task.tune(tune_option)
    print("tune done")


    print(f"start opitimize matmul, shape: input - [{BATCH},{NUM},{SQUEEZE},{IN}], mask - [{BATCH},{NUM},{SQUEEZE},{MASK}]")
    task = auto_scheduler.SearchTask(func=matmul,
                                    args=(BATCH, NUM, SQUEEZE, IN, MASK),
                                    target=target,
                                    layout_rewrite_option = auto_scheduler.LayoutRewriteOption.INSERT_TRANSFORM_STAGE) 
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "./result/matmul.json" 
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100000, 
        early_stopping = 500,     
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2, 
    )

    task.tune(tune_option)
    print("tune done")
