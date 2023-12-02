import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from thop import profile
from params import args
from performer import (
    EcoformerFastAttention,
    EcoformerSelfAttention,
    PerformerSelfAttention,
    EcoformerQuant
)
from performer_new import PerformerQuantAttention, FastQuantQK, FastQuantShareQK
from visual_distr import visual_distr
import deepshift.utils as utils
from fmoe_mlp import Shift_Linear
# from fmoe.gates import NaiveGate, NoisyGate
from fmoe_fc import Mlp_FMoE, PyTorchFMoE_FC
from fmoe_new import SparseDispatcher, NaiveGate
from matkernel import MatMul, MatAdd
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        linear=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.drop = drop
        self.linear = linear

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)
        self.B = 0
        self.N = 0 
        self.C = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        self.shape = []
        self.H = H
        self.W = W
        self.shape.append(x.shape)
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        self.shape.append(x.shape)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        self.shape.append(x.shape)
        x = self.fc2(x)
        x = self.drop(x)
            
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
       
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionShareQK(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_qk = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qk = (
            self.to_qk(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.to_v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (qk @ qk.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SRAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=False,
        moe_attn=False
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if moe_attn:
            self.q  = PyTorchFMoE_FC(dim, dim, bias=qkv_bias)
            self.kv = PyTorchFMoE_FC(dim, dim * 2, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = PyTorchFMoE_FC(dim, dim)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = (
                    self.kv(x_)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LinAngularAttention_binary(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sr_ratio=1,
        linear=False,
        sparse_reg=False,
        moe_attn=False
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        # self.q_quant = FastQuantQK(
        #     head_dim, None, generalized_attention=False,
        #     kernel_fn=nn.ReLU(), no_projection=False
        # )
        # self.k_quant = FastQuantQK(
        #     head_dim, None, generalized_attention=False,
        #     kernel_fn=nn.ReLU(), no_projection=False
        # )
        if moe_attn:
            self.qkv = PyTorchFMoE_FC(dim, dim * 3, bias=qkv_bias, gate=NaiveGate)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if moe_attn:
            self.proj = PyTorchFMoE_FC(dim, dim, gate=NaiveGate)
        else:
            self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatAdd()
        self.kqv_matmul = MatAdd()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # B, N, C = x.shape
        # q = (
        #     self.q(x)
        #     .reshape(B, N, self.num_heads, C // self.num_heads)
        #     .permute(0, 2, 1, 3)
        # )
        # kv = (
        #         self.kv(x)
        #         .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        #         .permute(2, 0, 3, 1, 4)
        #     )
        # k, v = kv[0], kv[1]

        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        # Quant
        binary_q_no_grad = torch.gt(q - q.mean(), 0).type(torch.float32)
        cliped_q = torch.clamp(q, 0, 1.0)
        q = binary_q_no_grad.detach() - cliped_q.detach() + cliped_q
        # s = q.size()
        # m = q.norm(p=1).div(q.nelement())
        # binary_q_no_grad = torch.gt(q, 0).type(torch.float32).mul(m.expand(s))
        # cliped_q = torch.clamp(q, 0, 1.0)
        # q = binary_q_no_grad.detach() - cliped_q.detach() + cliped_q

        binary_k_no_grad = torch.gt(k - k.mean(), 0).type(torch.float32)
        cliped_k = torch.clamp(k, 0, 1.0)
        k = binary_k_no_grad.detach() - cliped_k.detach() + cliped_k
        # s = k.size()
        # m = k.norm(p=1).div(k.nelement())
        # binary_k_no_grad = torch.gt(k, 0).type(torch.float32).mul(m.expand(s))
        # cliped_k = torch.clamp(k, 0, 1.0)
        # k = binary_k_no_grad.detach() - cliped_k.detach() + cliped_k

        dconv_v = self.dconv(v)

        attn = self.kq_matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                self.sv_matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinAngularAttention_ksh(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sr_ratio=1,
        linear=False,
        sparse_reg=False,
        moe_attn=False
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qk_quant = EcoformerQuant(
            num_heads,
            head_dim,
            nb_features=None,
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            no_projection=False,
        )

        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        if moe_attn:
            self.to_qk = PyTorchFMoE_FC(dim, dim, bias=qkv_bias)
            self.to_v  = PyTorchFMoE_FC(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = PyTorchFMoE_FC(dim, dim)
        else:
            self.to_qk = nn.Linear(dim, dim, bias=qkv_bias)
            self.to_v  = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
        # self.proj = FMoETransformerFC(2, dim, dim, world_size=1, top_k=1, gate=NaiveGate)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatAdd()
        self.kqv_matmul = MatAdd()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        qk = (
            self.to_qk(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.to_v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        qk = qk / qk.norm(dim=-1, keepdim=True)
        # TODO: add quant here
        qk = self.qk_quant(qk)
        # k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = self.kq_matmul(qk.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                self.sv_matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(qk, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.linear = linear
        # self.sr_ratio = sr_ratio
        # if not linear:
        #     if sr_ratio > 1:
        #         self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #         self.norm = nn.LayerNorm(dim)
        # else:
        #     self.pool = nn.AdaptiveAvgPool2d(7)
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        #     self.norm = nn.LayerNorm(dim)
        #     self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # if not self.linear:
        #     if self.sr_ratio > 1:
        #         x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        #         x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        #         x_ = self.norm(x_)
        #         kv = (
        #             self.kv(x_)
        #             .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        #             .permute(2, 0, 3, 1, 4)
        #         )
        #     else:
        #         kv = (
        #             self.kv(x)
        #             .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        #             .permute(2, 0, 3, 1, 4)
        #         )
        # else:
        #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        #     x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        #     x_ = self.norm(x_)
        #     x_ = self.act(x_)
        #     kv = (
        #         self.kv(x_)
        #         .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        #         .permute(2, 0, 3, 1, 4)
        #     )
        # k, v = kv[0], kv[1]
        kv = (
            self.kv(x)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        attn = (k.transpose(-2, -1) @ v) * self.scale

        x = (q @ attn).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


attn_dict = {
    "msa": Attention,
    "msa_share_qk": AttentionShareQK,
    "sra": SRAttention,
    "linear": LinearAttn,
    "performer": PerformerSelfAttention,
    "ecoformer": EcoformerSelfAttention,
    "performer_binary": PerformerQuantAttention,
    "LinAngular": LinAngularAttention_ksh
}


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        linear=False,
        last_stage=False,
        moe_mlp=False,
        moe_attn=False,
        world_size=0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if last_stage and args.attn_type!="sra":
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                linear=linear,
            )
        else:
            self.attn = attn_dict[args.attn_type](
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                linear=linear,
                moe_attn=moe_attn
            )

        # self.use_performer = use_performer
        # if use_performer:
        #     self.attn = PerformerSelfAttention(
        #         dim, num_heads=num_heads, qkv_bias=qkv_bias,
        #         qk_scale=qk_scale, attn_drop=attn_drop,
        #         proj_drop=drop, nb_features=args.num_features)
        # else:
        #     self.attn = Attention(dim,
        #         num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if not moe_mlp:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                linear=linear,
            )
        else:
            num_expert = 2
            if num_expert % world_size != 0:
                print("experts number of {} is not divisible by world size of {}".format(num_expert, world_size))
            gate = NaiveGate
            num_expert = num_expert // world_size
            self.mlp = Mlp_FMoE(
                d_model=dim,
                d_hidden=mlp_hidden_dim,
                activation=act_layer,
                drop=drop,
                linear=linear,
                gate=gate
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


# for Performer, start
def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class PyramidVisionTransformerV2(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        moe_mlp=False,
        moe_attn=False,
        world_size=1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.moe_mlp = moe_mlp
        self.moe_attn = moe_attn

        # self.use_performer = args.use_performer
        # if 1 in self.use_performer:
        #     self.feature_redraw_interval = 1
        #     self.register_buffer('calls_since_last_redraw', torch.tensor(0))

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        # if "rfa" in args.attn_type:
        #     self.auto_retrain = True
        #     self.feature_retrain_interval = args.k_iteration
        #     self.register_buffer("calls_since_last_retrain", torch.tensor(0))
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )
            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                        linear=linear,
                        last_stage=(i == num_stages - 1),
                        moe_mlp=moe_mlp,
                        moe_attn=moe_attn,
                        world_size=world_size,
                    )
                    for j in range(depths[i])
                ]
            )
            # block = nn.ModuleList([Block(
            #     dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
            #     qk_scale=qk_scale,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
            #     sr_ratio=sr_ratios[i], linear=linear, use_performer=self.use_performer[i]==1)
            #     for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = (
            nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def set_retrain(self):
        hash_attentions = find_modules(self, EcoformerFastAttention)
        for hash_attention in hash_attentions:
            hash_attention.is_trained = False

    def set_retrain_resume(self):
        hash_attentions = find_modules(self, EcoformerQuant)
        for hash_attention in hash_attentions:
            hash_attention.is_trained = True

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def set_loss(self, loss):
        self.balance_loss = loss

    def get_loss(self, clear=True):
        loss = self.balance_loss
        if clear:
            self.loss = None
        return loss

    def forward_features(self, x):
        B = x.shape[0]
        self.balance_loss = 0
        self.fc1_shift_ratio = []
        self.fc2_shift_ratio = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            
            for indx, blk in enumerate(block):
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    

    def plot_distribution(self):
        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            mlp = []
            attn = []
            for indx, blk in enumerate(block):
                mlp.append([blk.mlp.fc1.weight.data, blk.mlp.dwconv.dwconv.weight.data, blk.mlp.fc2.weight.data])
                attn.append([blk.attn.qkv.weight.data, blk.attn.proj.weight.data])
            visual_distr(mlp, 'mlp_stage'+str(i))
            visual_distr(attn, 'attn_stage'+str(i))


    def compute_FLOPs_diff(self, weight_bits, rounding):
        FLOPs = []
        diff = []
        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            for blk in block:
                FLOPs = blk.mlp.flops(FLOPs)
                diff = blk.mlp.convert_diff(diff, weight_bits, rounding)
        return FLOPs, diff
        # return diff

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvt_v2_b0(pretrained=False, moe_mlp=False, moe_attn=False, world_size=1, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        moe_mlp=moe_mlp,
        moe_attn=moe_attn,
        world_size=world_size,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def pvt_v2_b1(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b2(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b3(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b4(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b5(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b2_li(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        linear=True,
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model
