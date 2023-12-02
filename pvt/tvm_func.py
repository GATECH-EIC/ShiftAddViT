import time
import os
from collections import OrderedDict

import numpy as np
import torch
import tvm
from tvm import relay
from tvm import auto_scheduler
from tvm.contrib import graph_executor, pipeline_executor, pipeline_executor_build

from params import args
from logger import logger

def tvm_tune(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader, model_name:str = "pvt", device:int = 0):

    os.mkdir("./tvm_log", exist_ok=True)
    target = "cuda"
    # dev = tvm.device(target, 0)
    # load model    
    model = model.eval()
    data_loader_iter = iter(data_loader)
    input_data , trueValue = next(data_loader_iter) # get one batch data

    scripted_model = torch.jit.trace(model, input_data).eval()

    print("trace pytorch model done")
    # input pytorch model Graph to TVM relay
    input_name = "input"
    shape_list = [(input_name, input_data.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype = "float32")
    print(mod)

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    log_file = f"./tvm_log/{model_name}.json"

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=3,
                                                        min_repeat_ms=25,
                                                        timeout=15,
                                                        device=device)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=500000,
        early_stopping = 500,
        runner = measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)

    return

def tvm_throughput(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader, model_name:str = "pvt", device:int = 0):
    target = "cuda"
    dev = tvm.device(target, device)
    # load model    
    model = model.eval()
    data_loader_iter = iter(data_loader)
    input_data , trueValue = next(data_loader_iter) # get one batch data

    scripted_model = torch.jit.trace(model, input_data).eval()

    print("trace pytorch model done")
    # input pytorch model Graph to TVM relay
    input_name = "input"
    shape_list = [(input_name, input_data.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype = "float32")
    print(mod)

    log_file = f"./tvm_log/{model_name}.json"
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            # lib = relay.vm.compile(mod, target=relay.vm.Target(target=target), params=params)
            lib = relay.build(mod, target=target, params=params)

    print("=> build done")
    tvmModel = graph_executor.GraphModule(lib["default"](dev))
    input_data , trueValue = next(data_loader_iter) # get one batch data
    # Set inputs
    tvmModel.set_input(input_name, tvm.nd.array(input_data))

    # =========  thoughtput test ===============

    # warm up
    if args.batch_size == 1:
        repeat = 500
    else:
        repeat = 100

    for _ in range(repeat):
        tvmModel.run()
    logger.info(f"throughput averaged with {repeat} times")
    tic1 = time.time()
    for i in range(repeat):
        tvmModel.run()
    tic2 = time.time()
    logger.info(
        f"batch_size {args.batch_size} throughput {repeat * args.batch_size / (tic2 - tic1)}"
    )
    logger.info(
        f"batch_size {args.batch_size} latency {(tic2 - tic1) / repeat * 1000} ms"
    )  
    return