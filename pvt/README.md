
# Code for ShiftAddViT on PVT
<!-- ## 1. Handcraft replace linear/conv with shift layers

```
bash scripts/finetune.sh

# PoT-PS (PVT-Tiny)
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1246 \
    --use_env main.py \
    --config configs/pvt/pvt_tiny.py \
    --batch-size 32 \
    --data-path /home/shihuihong/imagenet \
    --data-set IMNET \
    --epochs 30 \
    --lr 1e-6 \
    --warmup-lr 5e-9 \
    --min-lr 5e-8 \
    --output_dir checkpoints/pvt_tiny_msa_shift \
    --finetune pvt_tiny.pth  \
    --shift_training \
    --shift_type 'PS' 
```
--shift_training: will partially convert layers in the original model with shift layers via the "convert_to_shift" function in line #488 of main.py

--shift_type: the construct mode of shift layers ('PS' or 'Q')

Attn:

1. Need to modify codes in line `#279` of `main.py` (for loading ckpt from pretrained model) if keys in ckpt are not matched with those in the constructed model

2. Need to modify codes in line `#30` of `deepshift/convert.py` to select which layers to be converted to the corresponding shift layers -->

## Training 

1. Train a PVT model (e.g., PVTv2 B0) with standard self-attention under 100 epochs. The model is initialized with corresponding pre-trained models in [PVT](https://github.com/whai362/PVT/tree/v2/classification).
```bash
# train with 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_msa.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --epochs 100 \
    --lr 5e-5 \
    --warmup-lr 1e-7 \
    --min-lr 1e-6 \
    --finetune [path of pvt_v2 pre-trained models] \
    --output_dir [output path of msa finetuned models] \
```

2. Convert MSA to linear attention and reparameterize all MatMuls with add layers 
```bash
# train with 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_LinAngular.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --epochs 100 \
    --lr 5e-5 \
    --warmup-lr 1e-7 \
    --min-lr 1e-6 \
    --finetune [path of msa finetuned models (from step 1)] \
    --output_dir [output path of finetuned models with linear attention where all MatMuls are replaced with add layers] \
```

3. Reparameterize MLPs and linear projection layers in attention with MoE layers
```bash
# train with 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_LinAngular.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --epochs 100 \
    --lr 1e-5 \
    --warmup-lr 5e-8 \
    --min-lr 5e-7 \
    --moe_attn \
    --moe_mlp \
    --finetune [path of finetuned models from step 2] \
    --output_dir [output path of ShiftAddViT models] \
```

## Evaluation

To evaluate a model, you can

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_LinAngular.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --resume [path of finetuned ShiftAddViT modelss] \
    --moe_attn \
    --moe_mlp \
    --eval
```

## TVM tune

To speedup model with TVM kernel, you can

```bash
python main.py \
    --config configs/pvt_v2/pvt_v2_b0_LinAngular.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --resume [path of finetuned ShiftAddViT modelss] \
    --moe_attn \
    --moe_mlp \
    --tvm_tune
```

## Latency test

To test latency on pytorch:
```bash
python main.py \
    --config configs/pvt_v2/pvt_v2_b0_LinAngular.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --resume [path of finetuned ShiftAddViT modelss] \
    --moe_attn \
    --moe_mlp \
    --throughput
```

To test latency with tuned TVM model (run TVM tune first):

```bash
python main.py \
    --config configs/pvt_v2/pvt_v2_b0_LinAngular.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --resume [path of finetuned ShiftAddViT modelss] \
    --moe_attn \
    --moe_mlp \
    --tvm_throughput
```


## Energy analyse

To analyse the energy cost of model:

```bash
python main.py \
    --config configs/pvt_v2/pvt_v2_b0_LinAngular.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --resume [path of finetuned ShiftAddViT modelss] \
    --moe_attn \
    --moe_mlp \
    --cal_energy
```