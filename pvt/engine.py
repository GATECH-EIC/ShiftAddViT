# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import time
import torch
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import utils
from logger import logger
from losses import DistillationLoss


def train_one_epoch(
    model: torch.nn.Module,
    criterion: DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    fp32=False,
):
    model.train(set_training_mode)
    # model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 400
    indx = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        indx += 1
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # with torch.cuda.amp.autocast():
        #     outputs = model(samples)
        #     loss = criterion(samples, outputs, targets)
        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)
            # print(outputs)
            loss = criterion(samples, outputs, targets)
            # try:
            #     balance_loss = model.module.get_loss()
            #     gate_ratio = model.module.gate_ratio
            #     # fc2_shift_ratio = model.module.fc2_shift_ratio
            # except:
            #     balance_loss = model.get_loss()
            #     gate_ratio = model.gate_ratio
                # fc2_shift_ratio = model.fc2_shift_ratio
            # print(balance_loss)
            balance_loss = 0
            if balance_loss != 0:
                loss_coef = 0.01
                total_loss = loss + loss_coef*balance_loss
            else:
                total_loss = loss
        
        # if balance_loss != 0:
        #     loss_value = loss.item() + balance_loss.item()
        # else:
        loss_value = total_loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            total_loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )
        # total_loss.backward()
        # optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # if balance_loss != 0 and indx % 1600 == 0:
        #     print("The gate ratio: ", gate_ratio)
            # print("The shift ratio for FC2: ", fc2_shift_ratio)
            # break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    # torch.cuda.set_device(0)
    # switch to evaluation mode
    model.eval()
    end = time.time()
    global_gate_ratio = []
    global_fc2_shift_ratio = []
    indx = 0
    for images, target in metric_logger.log_every(data_loader, 400, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        indx += 1
        # images = images.cuda()
        # target = target.cuda()

        # import numpy as np
        # np.save("./visualization/image.npy", images.cpu())
        # exit()

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["time"].update(time.time() - end)
        if model.module.get_loss()!=0 and indx % 400==0:
            print("The gate ratio: ", model.module.gate_ratio)
            # print("The shift ratio for FC2: ", model.module.fc2_shift_ratio)
            # break
        end = time.time()
        # indx += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} time {time.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, time=metric_logger.time
        )
    )
    # global_gate_ratio = [global_gate_ratio[i]/indx for i in range(len(global_gate_ratio))]
    # global_fc2_shift_ratio = [global_fc2_shift_ratio[i]/indx for i in range(len(global_fc2_shift_ratio))]
    # print("The shift ratio for FC1: ", global_gate_ratio)
    # print("The shift ratio for FC2: ", global_fc2_shift_ratio)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
