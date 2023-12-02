import json
import os
import time
import warnings
from datetime import timedelta
from pathlib import Path
import fmoe
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict
from deepshift.convert import convert_to_shift, convert_to_multi
# import models
import pvt
import pvt_v2
import utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from logger import logger
from losses import DistillationLoss
from params import args
from pvt_v2 import Attention
from samplers import RASampler
import deepshift
# from quantize_hm.quantize import qconfig
warnings.filterwarnings("ignore")
from gpu_mem_track import MemTracker
from unoptimized.convert import convert_to_unoptimized
# from accelerate import Accelerator

from tvm_func import tvm_tune, tvm_throughput
from cal_energy import cal_modelEnergy

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(100):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 100 times")
        tic1 = time.time()
        for i in range(100):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {100 * batch_size / (tic2 - tic1)}"
        )
        logger.info(
            f"batch_size {batch_size} latency {(tic2 - tic1) / 100 * 1000} ms"
        )
        return

def sync_weights(model, except_key_words):
    state_dict = model.state_dict()
    for key, item  in state_dict.items():
        flag_sync = True
        for key_word in except_key_words:
            if key_word in key:
                # print('key',key)
                flag_sync = False
                break

        if flag_sync:
            torch.distributed.broadcast(item, 0)

    model.load_state_dict(state_dict)
    return

def main():
    utils.init_distributed_mode(args)

    if utils.get_rank() != 0:
        logger.disabled = True

    logger.info(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if args.tvm_tune or args.tvm_throughput:
        args.batch_size = 1

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank,
                shuffle=True,
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank,
                shuffle=False,
            )
        else:
            # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_val = torch.utils.data.RandomSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        # batch_size=int(1.5 * args.batch_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    logger.info(f"Creating model: {args.model}")
    
    world_size = 1
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        moe_attn=args.moe_attn,
        moe_mlp=args.moe_mlp,
        world_size= world_size
    )
    

    def load_experts(k, checkpoint_model):
        if "experts.0" in k:
            new_k = k.replace('experts.0.', '')
        elif "experts.1" in k:
            new_k = k.replace('experts.1.', '')
        elif "experts.2" in k:
            new_k = k.replace('experts.2.', '')
        elif "experts.3" in k:
            new_k = k.replace('experts.3.', '')
        
        try:
            ckpt = checkpoint_model[new_k].float()
        except: 
            try:
                if 'shift' in k:
                    new_k_2 = new_k.replace('shift', 'weight')
                elif 'sign' in k:
                    new_k_2 = new_k.replace('sign', 'weight')
                ckpt = checkpoint_model[new_k_2].float()
            except:
                return None, None
        return new_k, ckpt


    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint
        
        state_keys = list(checkpoint_model.keys())
        # print(state_keys)
        for k, p in model.named_parameters():
            # print(k)
            if k in checkpoint_model:
                state_keys.remove(k)
                shape_original = p.shape
                p.data = checkpoint_model[k].float()
                shape_new = p.shape
                if shape_original != shape_new:
                    print(k)
            else:
                if args.moe_attn or args.moe_mlp:
                    if "experts" in k:
                        new_k, ckpt = load_experts(k, checkpoint_model)
                    if "experts.0" in k:
                        shape_original = p.shape
                        p.data = ckpt
                        state_keys.remove(new_k)
                        shape_new = p.shape
                        if shape_original != shape_new:
                            print(k)
                    elif "experts.1" in k: 
                        shape_original = p.shape
                        if 'bias' in k:
                            p.data = ckpt
                        elif 'shift' in k:
                            p.data, sign = deepshift.utils.get_shift_and_sign(ckpt) 
                        elif 'sign' in k:
                            shift, p.data = deepshift.utils.get_shift_and_sign(ckpt)
                        shape_new = p.shape
                        if shape_original != shape_new:
                            print(k)
                elif args.attn_type == "msa":
                    # load from pretrained pvtv2
                    refer_k = k.split(".")
                    refer_k[3] = "q"
                    target_key = ".".join(refer_k)
                    q_weight = checkpoint_model[target_key].float()
                    state_keys.remove(target_key)

                    refer_k[3] = "kv"
                    target_key = ".".join(refer_k)
                    kv_weight = checkpoint_model[target_key].float()
                    state_keys.remove(target_key)

                    p.data = torch.cat((q_weight, kv_weight), dim=0)
                elif args.attn_type in ["ecoformer"]:
                    # load from msa weight
                    refer_k = k.split(".")
                    source_k = refer_k[3]
                    refer_k[3] = "qkv"
                    target_key = ".".join(refer_k)
                    qkv_weight = checkpoint_model[target_key].float()
                    shapes = qkv_weight.shape
                    dim = shapes[0] // 3
                    if target_key in state_keys:
                        state_keys.remove(target_key)
                    if source_k == "to_qk":
                        p.data = qkv_weight[:dim, ...]
                    elif source_k == "to_v":
                        p.data = qkv_weight[-dim:, ...]
                elif args.attn_type in ["LinAngular"]:
                    # load from msa weight
                    # refer_k = k.split(".")
                    # source_k = refer_k[3]
                    # if source_k == "to_qk":
                    #     refer_k[3] = "q"
                    #     target_key = ".".join(refer_k)
                    #     qkv_weight = checkpoint_model[target_key].float()
                    #     shapes = qkv_weight.shape
                    #     dim = shapes[0] 
                    # elif source_k == "to_v":
                    #     refer_k[3] = "kv"
                    #     target_key = ".".join(refer_k)
                    #     qkv_weight = checkpoint_model[target_key].float()
                    #     shapes = qkv_weight.shape
                    #     dim = shapes[0] // 2
                    
                    # if target_key in state_keys:
                    #     state_keys.remove(target_key)
                    # if source_k == "to_qk":
                    #     p.data = qkv_weight
                    # elif source_k == "to_v":
                    #     p.data = qkv_weight[-dim:, ...]
                    # -------------------------------------------------
                    # load from msa weight
                    # refer_k = k.split(".")
                    # source_k = refer_k[3]
                    # refer_k[3] = "qkv"
                    # target_key = ".".join(refer_k)
                    # qkv_weight = checkpoint_model[target_key].float()
                    # shapes = qkv_weight.shape
                    # dim = shapes[0] // 3
                    # if target_key in state_keys:
                    #     state_keys.remove(target_key)
                    # if source_k == "to_qk":
                    #     p.data = qkv_weight[:dim, ...]
                    # elif source_k == "to_v":
                    #     p.data = qkv_weight[-dim:, ...]
                    pass

        if len(state_keys) > 0:
            for temp_key in state_keys:
                logger.info("Not used: %s" % temp_key)

        # state_dict = model.state_dict()
        # for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         logger.info(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
        #
        # model.load_state_dict(checkpoint_model, strict=False)
        # ##################################
        # model.set_retrain_resume()
    # model.plot_distribution()
    # exit()
    model.cuda()
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"number of params: {n_parameters}")

    model_ema = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    logger.info(str(model_without_ddp))

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    criterion = DistillationLoss(criterion, None, "none", 0, 0)

    output_dir = Path(args.output_dir)
    
    # linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr
    # optimizer = create_optimizer(args, model_without_ddp)
    # loss_scaler = NativeScaler()
    # lr_scheduler, _ = create_scheduler(args, optimizer)

    # TODO: convert conv/linear to shift 
    if args.device == 'cuda':
        use_cuda = True
    else:
        use_cuda = False
    if args.shift_training:
        act_integer_bits=16 
        act_fraction_bits=16
        shift_type=args.shift_type
        weight_bits = 5
        rounding = 'deterministic'
        if args.distributed:
            model.module, conversion_count, linear_count, conv_count = convert_to_shift(model_without_ddp, shift_depth=100, shift_type=shift_type, convert_all_linear=True, convert_weights=True, freeze_sign=False, use_kernel=args.use_kernel, use_cuda=use_cuda, rounding=rounding, weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits, SP2=args.SP2)
            model_without_ddp = model.module
            logger.info(str(model.module))
        else:
            model, conversion_count, linear_count, conv_count = convert_to_shift(model_without_ddp, shift_depth=100, shift_type=shift_type, convert_all_linear=True, convert_weights=True, freeze_sign=False, use_kernel=args.use_kernel, use_cuda=use_cuda, rounding=rounding, weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits, SP2=args.SP2)
            model_without_ddp = model
            logger.info(str(model))
        shift_flag = True
    elif args.use_kernel:
        if args.distributed:
            model.module, linear_count, conv_count = convert_to_unoptimized(model.module)
        else:
            model, linear_count, conv_count = convert_to_unoptimized(model)
    
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        if "model" in checkpoint:
            msg = model_without_ddp.load_state_dict(checkpoint["model"])
        else:
            msg = model_without_ddp.load_state_dict(checkpoint)
        logger.info(str(msg))
        # if (
        #     (not args.eval or args.shift_training)
        #     and "optimizer" in checkpoint
        #     and "lr_scheduler" in checkpoint
        #     and "epoch" in checkpoint
        # ):
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        #     args.start_epoch = checkpoint["epoch"] + 1
        #     # if args.model_ema:
        #     #     utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        #     if "scaler" in checkpoint:
        #         loss_scaler.load_state_dict(checkpoint["scaler"])
        if not args.shift_training:
            if "epoch" in checkpoint:
                if hasattr(model, "module"):
                    model.module.set_retrain_resume()
                else:
                    model.set_retrain_resume()
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            # if args.model_ema:
            #     utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
        if "epoch" in checkpoint:
            if hasattr(model, "module"):
                model.module.set_retrain_resume()
            else:
                model.set_retrain_resume()


    if args.tvm_tune:
        tvm_tune(model, data_loader_val)
        return

    if args.tvm_throughput:
        tvm_throughput(model, data_loader_val)
        return

    if args.throughput:
        throughput(data_loader_val, model, logger)
        return
    
    if args.cal_energy:
        cal_modelEnergy(model, data_loader_val)
        return
    
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    gpu_tracker = MemTracker()
    # print("Start Epochs: ", args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        # if 1 in args.use_performer:
        #     if hasattr(model, 'module'):
        #         model.module.feature_redraw_interval = 1 + 5 * epoch
        #     else:
        #         model.feature_redraw_interval = 1 + 5 * epoch
        # if hasattr(args, "k"):
        #     if epoch % args.k == 0:
        #         if hasattr(model, "module"):
        #             model.module.set_retrain()
        #         else:
        #             model.set_retrain()

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # model, optimizer, data_loader_train = accelerator.prepare(model, optimizer, data_loader_train)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.finetune
            == "",  # keep in eval mode during finetuning
            fp32=args.fp32_resume,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "last_checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(data_loader_val, model, device)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        if max_accuracy < test_stats["acc1"]:
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "scaler": loss_scaler.state_dict(),
                    "args": args,
                },
                os.path.join(args.output_dir, "best_checkpoint.pth"),
            )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    main()
