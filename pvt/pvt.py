from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fmoe_mlp import PyTorchFMoE_MLP
from fmoe_fc import PyTorchFMoE_FC
from params import args
# from fmoe.gates import NaiveGate, NoisyGate
from performer import EcoformerQuant
from hashing.utils import BinaryQuantizer
from performer import (
    EcoformerFastAttention,
    EcoformerSelfAttention,
    PerformerSelfAttention,
    EcoformerQuant
)
from fmoe_new import SparseDispatcher, NaiveGate, NoisyGate, NoisyGate_V2
from matkernel import MatMul, MatAdd

__all__ = ["pvt_tiny", "pvt_small", "pvt_medium", "pvt_large"]


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class LowHighFreAttention(nn.Module):
    """
    low high frequency, with alpha to split heads
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=2,
        alpha=0.5,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.l_heads = int(num_heads * alpha)
        self.h_heads = num_heads - self.l_heads

        head_dim = int(dim / num_heads)
        self.dim = dim
        self.l_dim = self.l_heads * head_dim
        self.h_dim = self.h_heads * head_dim
        self.sr_ratio = window_size
        assert window_size > 1
        self.scale = qk_scale or head_dim ** -0.5

        if self.l_heads > 0:
            # for low frequency
            self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)
            self.l_proj_drop = nn.Dropout(proj_drop)
            self.l_attn_drop = nn.Dropout(attn_drop)

        # for high frequency
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)
            self.l_proj_drop = nn.Dropout(proj_drop)
            self.l_attn_drop = nn.Dropout(attn_drop)

        self.ws = window_size

    def extract_high_freq(self, x):
        # B, N, C = x.shape
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        window_mean = x.reshape(B, total_groups, -1, C).mean(
            dim=2
        )  # mean is low frequency info
        window_mean = (
            window_mean.reshape(B, h_group, w_group, C)
            .unsqueeze(3)
            .unsqueeze(4)
            .expand_as(x)
        )
        x = x - window_mean

        qkv = (
            self.h_qkv(x)
            .reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads)
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.h_attn_drop(
            attn
        )  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (
            (attn @ v)
            .transpose(2, 3)
            .reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        )
        x = attn.transpose(2, 3).reshape(
            B, h_group * self.ws, w_group * self.ws, self.h_dim
        )
        x = self.h_proj(x)
        x = self.h_proj_drop(x)
        return x

    def extract_low_freq(self, x):
        # B, N, C = x.shape
        B, H, W, C = x.shape

        q = (
            self.l_q(x)
            .reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = (
                self.l_kv(x_)
                .reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.l_attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        x = self.l_proj_drop(x)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape
        # H = W = int(N ** 0.5)

        x = x.reshape(B, H, W, C)
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        if self.h_heads == 0:
            # only processing low frequency
            x = self.extract_low_freq(x)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]
            return x.reshape(B, N, C)

        if self.l_heads == 0:
            # only processing high frequency, local window attention
            x = self.extract_high_freq(x)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]
            return x.reshape(B, N, C)

        # both low and high frequency
        high_attn = self.extract_high_freq(x)
        low_attn = self.extract_low_freq(x)
        if pad_r > 0 or pad_b > 0:
            x = torch.cat((high_attn[:, :H, :W, :], low_attn[:, :H, :W, :]), dim=-1)
        else:
            x = torch.cat((high_attn, low_attn), dim=-1)
        x = x.reshape(B, N, C)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LinAngularAttention_binary(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
        moe_attn=False,
        qk_scale=None,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if moe_attn:
            self.q  = PyTorchFMoE_FC(dim, dim, bias=qkv_bias)
            self.kv = PyTorchFMoE_FC(dim, dim * 2, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if moe_attn:
            self.proj = PyTorchFMoE_FC(dim, dim)
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
        # N, L, C = x.shape
        # qkv = (
        #     self.qkv(x)
        #     .reshape(N, L, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        # Quant
        # binary_q_no_grad = torch.gt(q - q.mean(), 0).type(torch.float32)
        # cliped_q = torch.clamp(q, 0, 1.0)
        # q = binary_q_no_grad.detach() - cliped_q.detach() + cliped_q
        # q = BinaryQuantizer.apply(q - q.mean())
        s = q.size()
        m = q.norm(p=1).div(q.nelement())
        binary_q_no_grad = torch.gt(q, 0).type(torch.float32).mul(m.expand(s))
        cliped_q = torch.clamp(q, 0, 1.0)
        q = binary_q_no_grad.detach() - cliped_q.detach() + cliped_q

        # binary_k_no_grad = torch.gt(k - k.mean(), 0).type(torch.float32)
        # cliped_k = torch.clamp(k, 0, 1.0)
        # k = binary_k_no_grad.detach() - cliped_k.detach() + cliped_k
        # k = BinaryQuantizer.apply(k - k.mean())
        s = k.size()
        m = k.norm(p=1).div(k.nelement())
        binary_k_no_grad = torch.gt(k, 0).type(torch.float32).mul(m.expand(s))
        cliped_k = torch.clamp(k, 0, 1.0)
        k = binary_k_no_grad.detach() - cliped_k.detach() + cliped_k
        
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
        self.moe_attn = moe_attn
        
        self.qk_quant = EcoformerQuant(
            num_heads,
            head_dim,
            nb_features=None,
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            no_projection=False,
        )
        # self.qk_quant = FastQuantShareQK(
        #     head_dim, None, generalized_attention=False,
        #     kernel_fn=nn.ReLU(), no_projection=False
        # )

        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg
        if self.moe_attn:
            self.to_qk = PyTorchFMoE_FC(dim, dim, bias=qkv_bias, gate=NaiveGate)
            self.to_v =  PyTorchFMoE_FC(dim, dim, bias=qkv_bias, gate=NaiveGate)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = PyTorchFMoE_FC(dim, dim, gate=NaiveGate)
            self.proj_drop = nn.Dropout(proj_drop)
        else:
            self.to_qk = nn.Linear(dim, dim, bias=qkv_bias)
            self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
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

    def forward(self, x, H, W, attn_shift_ratio=None):
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

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # print(torch.isnan(x).any())
        # print(torch.isnan(self.qkv.weight).any())
        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # q, k, v = (
        #     qkv[0],
        #     qkv[1],
        #     qkv[2],
        # )  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
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

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

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
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


attn_dict = {
    "msa": Attention,
    "sra": SRAttention,
    "performer": PerformerSelfAttention,
    "ecoformer": EcoformerSelfAttention,
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
        use_performer=False,
        # moe=False,
        world_size=0,
        linear_attn=False,
        last_stage=False,
        moe_mlp=False,
        moe_attn=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.moe_mlp = moe_mlp
        self.moe_attn = moe_attn
        self.last_stage = last_stage
        # self.use_performer = use_performer
        if last_stage:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
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
                moe_attn=moe_attn
            )

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
            )
        else:
            num_expert = 2
            if num_expert % world_size != 0:
                print("experts number of {} is not divisible by world size of {}".format(num_expert, world_size))
            num_expert = num_expert // world_size
            
            gate = NaiveGate
            self.mlp = PyTorchFMoE_MLP(
                num_expert=2,
                d_model=dim,
                d_hidden=mlp_hidden_dim,
                activation=act_layer,
                drop=drop,
                linear=False,
                world_size=world_size,
                top_k=1,
                gate=gate
            )

    def forward(self, x, H, W, shift_ratio=None):
        
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
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
        stage3_alpha=0.5,
        # moe=False,
        world_size=1,
        linear_attn=False,
        moe_mlp=False,
        moe_attn=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.moe_mlp = moe_mlp
        self.moe_attn = moe_attn
        # self.use_performer = args.use_performer
        self.shift_ratio = []
        self.shift_ratio_observed = []
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        alphas = [None, 0.5, stage3_alpha, None]
        # alphas = [None] * 4
        for i in range(num_stages):
            patch_embed = PatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=patch_size if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )
            num_patches = (
                patch_embed.num_patches
                if i != num_stages - 1
                else patch_embed.num_patches + 1
            )
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

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
                        world_size=world_size,
                        last_stage=(i == num_stages - 1),
                        moe_mlp=moe_mlp,
                        moe_attn=moe_attn,
                    )
                    for j in range(depths[i])
                ]
            )
            # block = nn.ModuleList([Block(
            #     dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
            #     qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
            #     norm_layer=norm_layer, sr_ratio=sr_ratios[i], use_performer=self.use_performer[i]==1)
            #     for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = (
            nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        )

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {"cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def gate_observer_open(self):
        MoEs = find_modules(self, PyTorchFMoE_MLP)
        for moe in MoEs:
            moe.gate.observe = True
        MoEs = find_modules(self, PyTorchFMoE_FC)
        for moe in MoEs:
            moe.gate.observe = True

    def gate_observer_close(self):
        MoEs = find_modules(self, PyTorchFMoE_MLP)
        for moe in MoEs:
            moe.gate.observe = False
        MoEs = find_modules(self, PyTorchFMoE_FC)
        for moe in MoEs:
            moe.gate.observe = False

    def set_retrain_resume(self):
        hash_attentions = find_modules(self, EcoformerQuant)
        for hash_attention in hash_attentions:
            hash_attention.is_trained = True

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return (
                F.interpolate(
                    pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(
                        0, 3, 1, 2
                    ),
                    size=(H, W),
                    mode="bilinear",
                )
                .reshape(1, -1, H * W)
                .permute(0, 2, 1)
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
        self.gate_ratio = []
        self.balance_loss = 0
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for k, blk in enumerate(block):
                x = blk(x, H, W)
                if self.moe_mlp and (blk.mlp.gate_type==NoisyGate):
                    self.gate_ratio.append(blk.mlp.gate.shift_ratio)
                    self.balance_loss += blk.mlp.gate.get_loss()
            if (self.moe_mlp or self.moe_attn) and (blk.mlp.gate_type==NoisyGate):
                self.set_loss(self.balance_loss)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

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
def pvt_tiny(pretrained=False, moe_mlp=False, moe_attn=False, world_size=1, linear_attn=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        moe_mlp=moe_mlp,
        moe_attn=moe_attn,
        world_size=world_size,
        linear_attn=linear_attn,
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        moe_mlp = False,
        moe_attn = False,
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
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
def pvt_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
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
def pvt_huge_v2(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4,
        embed_dims=[128, 256, 512, 768],
        num_heads=[2, 4, 8, 12],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 10, 60, 3],
        sr_ratios=[8, 4, 2, 1],
        # drop_rate=0.0, drop_path_rate=0.02)
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model
