import argparse
import os
from pathlib import Path

import mmcv

parser = argparse.ArgumentParser("PVT training and evaluation script", add_help=False)
parser.add_argument("--fp32-resume", action="store_true", default=False)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--config", required=True, type=str, help="config")

# Model parameters
parser.add_argument(
    "--model",
    default="pvt_small",
    type=str,
    metavar="MODEL",
    help="Name of model to train",
)
parser.add_argument("--input-size", default=224, type=int, help="images input size")

parser.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=0.1,
    metavar="PCT",
    help="Drop path rate (default: 0.1)",
)

# parser.add_argument('--model-ema', action='store_true')
# parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
# parser.set_defaults(model_ema=True)
# parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
# parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="adamw",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "adamw"',
)
parser.add_argument(
    "--opt-eps",
    default=1e-8,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: 1e-8)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
)
# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="cosine",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "cosine"',
)
parser.add_argument(
    "--lr", type=float, default=5e-5, metavar="LR", help="learning rate (default: 5e-4)"
)
parser.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
parser.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=1e-7,
    metavar="LR",
    help="warmup learning rate (default: 1e-6)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-6,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)

parser.add_argument(
    "--decay-epochs",
    type=float,
    default=30,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation parameters
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default="rand-m9-mstd0.5-inc1",
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". " + \
                         "(default: rand-m9-mstd0.5-inc1)',
),
parser.add_argument(
    "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
)
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="bicubic",
    help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
)

parser.add_argument("--repeated-aug", action="store_true")
parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
parser.set_defaults(repeated_aug=True)

# * Random Erase params
parser.add_argument(
    "--reprob",
    type=float,
    default=0.25,
    metavar="PCT",
    help="Random erase prob (default: 0.25)",
)
parser.add_argument(
    "--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")'
)
parser.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)

# * Mixup params
parser.add_argument(
    "--mixup",
    type=float,
    default=0.8,
    help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=1.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
)
parser.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Probability of performing mixup or cutmix when either/both is enabled",
)
parser.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Probability of switching to cutmix when both mixup and cutmix enabled",
)
parser.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
)

# Distillation parameters
# parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
#                     help='Name of teacher model to train (default: "regnety_160"')
# parser.add_argument('--teacher-path', type=str, default='')
# parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
# parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
# parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

# * Finetuning params
parser.add_argument("--finetune", default="", help="finetune from checkpoint")

# Dataset parameters
parser.add_argument(
    "--data-path",
    default="/datasets01/imagenet_full_size/061417/",
    type=str,
    help="dataset path",
)
parser.add_argument(
    "--data-set", default="IMNET", type=str, help="Image Net dataset path"
)
parser.add_argument(
    "--use-mcloader", action="store_true", default=False, help="Use mcloader"
)
parser.add_argument(
    "--inat-category",
    default="name",
    choices=[
        "kingdom",
        "phylum",
        "class",
        "order",
        "supercategory",
        "family",
        "genus",
        "name",
    ],
    type=str,
    help="semantic granularity",
)

parser.add_argument(
    "--output_dir", default="", help="path where to save, empty for no saving"
)
parser.add_argument(
    "--device", default="cuda", help="device to use for training / testing"
)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--resume", default="", help="resume from checkpoint")
parser.add_argument(
    "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
)
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--shift_training", action="store_true", help="Start the training of shift layers")
parser.add_argument("--moe_attn", action="store_true", help="Replace the vanilla projections lin attention with MoE")
parser.add_argument("--moe_mlp", action="store_true", help="Replace the vanilla MLP with MoE")
parser.add_argument("--moe_data_distributed", action="store_true", help="Use data parallel or model parallel to train moe")
parser.add_argument("--progressive_training", action="store_true", help="leverage the progressive training to train shift layers")
parser.add_argument("--SP2", action="store_true", help="Use SP2 to train shift layers")
parser.add_argument("--shift_type", default="PS", help="the mode of shift layers")
parser.add_argument("--use_kernel", action="store_true", help="Use customized kernel or not")
parser.add_argument(
    "--dist-eval",
    action="store_true",
    default=False,
    help="Enabling distributed evaluation",
)
parser.add_argument("--num_workers", default=10, type=int)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
parser.set_defaults(pin_mem=True)

# distributed training parameters
parser.add_argument(
    "--world_size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument(
    "--dist_url", default="env://", help="url used to set up distributed training"
)
parser.add_argument("--throughput", action="store_true", help="Perform evaluation only")
parser.add_argument(
    "--stage3_alpha", default=0.5, type=float, help="number of distributed processes"
)
parser.add_argument(
    "--num_features", default=256, type=int, help="number of features for performer"
)
parser.add_argument("--attn_type", default="sra", type=str, help="attention type")
parser.add_argument("--nbits", default=16, type=int, help="number of bits")
# parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

parser.add_argument("--tvm_tune", action="store_true", help="speed up model with TVM")
parser.add_argument("--tvm_throughput", action="store_true", help="test the thoughtput using TVM")

args = parser.parse_args()

cfg = mmcv.Config.fromfile(args.config)
for _, cfg_item in cfg._cfg_dict.items():
    for k, v in cfg_item.items():
        setattr(args, k, v)
if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# if args.resume == "":
#     checkpoint_path = os.path.join(args.output_dir, "last_checkpoint.pth")
#     if os.path.exists(checkpoint_path):
#         setattr(args, "resume", checkpoint_path)

