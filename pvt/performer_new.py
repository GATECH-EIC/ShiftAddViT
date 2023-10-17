# mainly modified from
# https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
import math
from scipy.stats import ortho_group
import torch
from torch import nn
from einops import rearrange, repeat

from functools import partial
from params import args

# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel_share_qk(data, *, projection_matrix, normalize_data=True,
                   eps=1e-4, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data),
                             projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True,
                   eps=1e-4, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data),
                             projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data -
                          torch.max(data_dash, dim=-1,
                                    keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(),
                       kernel_epsilon=0.001, normalize_data=True, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data),
                             projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0,
                                      device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = torch.FloatTensor(ortho_group.rvs(nb_columns), device='cpu').to(device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = torch.FloatTensor(ortho_group.rvs(nb_columns), device='cpu').to(device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(
            dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,),
                                                                 device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# non-causal linear attention
def linear_attention_bias(q, k, v):
    B, H, N, D = q.shape
    bias = args.num_hashes + 1
    top_bias = v * bias
    bottom_bias = N * bias
    k_cumsum = k.sum(dim=-2).float()
    D_inv = 1. / (torch.einsum('...nd,...d->...n', q.float(), k_cumsum) + bottom_bias)
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd->...ne', context, q) + top_bias
    out2 = torch.einsum('...ne,...n->...ne', out, D_inv)
    return out2


def linear_attention_bias_quant(q, k, v):
    B, H, N, D = q.shape
    bias = D + 1
    top_bias = v * bias
    bottom_bias = N * bias
    with torch.cuda.amp.autocast(enabled=False):
        k_cumsum = k.sum(dim=-2).float()
        D_inv = 1. / (torch.einsum('...nd,...d->...n', q.float(), k_cumsum) + bottom_bias)
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd->...ne', context.float(), q.float()) + top_bias
        out2 = torch.einsum('...ne,...n->...ne', out, D_inv)
    return out2

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0,
                 generalized_attention=False, kernel_fn=nn.ReLU(),
                 no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix,
                                         nb_rows=self.nb_features,
                                         nb_columns=dim_heads,
                                         scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        self.__flops__ = 0

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel,
                                    kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix,
                                    device=device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel,
                                    projection_matrix=self.projection_matrix,
                                    device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        out = linear_attention(q, k, v)
        return out

    @staticmethod
    def compute_macs(module, input, output):
        input = input[0]
        _, H, N, C = input.shape
        Nf = module.nb_features
        assert C == module.dim_heads
        macs = 0
        n_params = 0

        if module.no_projection:
            raise ValueError("Not supported yet!")
        elif module.generalized_attention:
            raise ValueError("Not supported yet!")
        else:
            n_params += C * Nf
            # q = create_kernel(q, is_query=True)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf
            # k = create_kernel(k, is_query=False)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf

        # out = linear_attention(q, k, v)
        # k_cumsum = k.sum(dim=-2)
        macs += H * N * Nf
        # D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        macs += H * N * Nf
        # context = torch.einsum('...nd,...ne->...de', k, v)
        macs += H * N * Nf * C
        # out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        macs += 2 * H * N * Nf * C
        # print('macs fast att', macs / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class FastQuantShareQK(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0,
                 generalized_attention=False, kernel_fn=nn.ReLU(),
                 no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix,
                                         nb_rows=self.nb_features,
                                         nb_columns=dim_heads,
                                         scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        # self.num_hashes = args.num_hashes
        self.__flops__ = 0

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, qk):
        device = qk.device

        create_kernel = partial(softmax_kernel_share_qk,
                                projection_matrix=self.projection_matrix,
                                device=device)
        qk = create_kernel(qk)


        # quantize qk
        # TODO:
        # binary_input_no_grad = torch.sign(qk - qk.mean())
        # cliped_input = torch.clamp(qk, -1.0, 1.0)
        binary_input_no_grad = torch.gt(qk - qk.mean(), 0).type(torch.float32)
        cliped_input = torch.clamp(qk, 0, 1.0)
        qk = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        # # hashing attention
        # B, H, N, D = qk.shape
        # rotaion_shape = (H, D, self.num_hashes, 1)
        # random_rotations = torch.randn(rotaion_shape, dtype=qk.dtype, device=qk.device)[None, ...].expand((B, H, D, self.num_hashes, 1)).reshape(B*H, D, self.num_hashes, 1)
        #
        # qk = qk.reshape(B*H, N, D)
        #
        # rotated_qk = torch.einsum('btf,bfhi->bhti', qk, random_rotations).squeeze(dim=-1).permute(0, 2, 1)
        # binary_input_no_grad = torch.sign(rotated_qk)
        # cliped_input = torch.clamp(rotated_qk, -1.0, 1.0)
        # qk = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        # qk = qk.reshape(B, H, N, self.num_hashes)

        # out = linear_attention_bias_quant(qk, qk, v)
        return qk

    @staticmethod
    def compute_macs(module, input, output):
        input = input[0]
        _, H, N, C = input.shape
        Nf = module.nb_features
        assert C == module.dim_heads
        macs = 0
        n_params = 0

        if module.no_projection:
            raise ValueError("Not supported yet!")
        elif module.generalized_attention:
            raise ValueError("Not supported yet!")
        else:
            n_params += C * Nf
            # q = create_kernel(q, is_query=True)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf
            # k = create_kernel(k, is_query=False)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf

        # out = linear_attention(q, k, v)
        # k_cumsum = k.sum(dim=-2)
        macs += H * N * Nf
        # D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        macs += H * N * Nf
        # context = torch.einsum('...nd,...ne->...de', k, v)
        macs += H * N * Nf * C
        # out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        macs += 2 * H * N * Nf * C
        # print('macs fast att', macs / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class FastQuantQK(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0,
                 generalized_attention=False, kernel_fn=nn.ReLU(),
                 no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix,
                                         nb_rows=self.nb_features,
                                         nb_columns=dim_heads,
                                         scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        # self.num_hashes = args.num_hashes
        self.__flops__ = 0

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, qk, is_query):
        device = qk.device

        create_kernel = partial(softmax_kernel,
                                projection_matrix=self.projection_matrix,
                                device=device)
        qk = create_kernel(qk, is_query=is_query)

        # quantize qk
        # TODO:
        # binary_input_no_grad = torch.sign(qk - qk.mean())
        # cliped_input = torch.clamp(qk, -1.0, 1.0)
        binary_input_no_grad = torch.gt(qk - qk.mean(), 0).type(torch.float32)
        cliped_input = torch.clamp(qk, 0, 1.0)
        qk = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        # # hashing attention
        # B, H, N, D = qk.shape
        # rotaion_shape = (H, D, self.num_hashes, 1)
        # random_rotations = torch.randn(rotaion_shape, dtype=qk.dtype, device=qk.device)[None, ...].expand((B, H, D, self.num_hashes, 1)).reshape(B*H, D, self.num_hashes, 1)
        #
        # qk = qk.reshape(B*H, N, D)
        #
        # rotated_qk = torch.einsum('btf,bfhi->bhti', qk, random_rotations).squeeze(dim=-1).permute(0, 2, 1)
        # binary_input_no_grad = torch.sign(rotated_qk)
        # cliped_input = torch.clamp(rotated_qk, -1.0, 1.0)
        # qk = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        # qk = qk.reshape(B, H, N, self.num_hashes)

        # out = linear_attention_bias_quant(qk, qk, v)
        return qk

    @staticmethod
    def compute_macs(module, input, output):
        input = input[0]
        _, H, N, C = input.shape
        Nf = module.nb_features
        assert C == module.dim_heads
        macs = 0
        n_params = 0

        if module.no_projection:
            raise ValueError("Not supported yet!")
        elif module.generalized_attention:
            raise ValueError("Not supported yet!")
        else:
            n_params += C * Nf
            # q = create_kernel(q, is_query=True)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf
            # k = create_kernel(k, is_query=False)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf

        # out = linear_attention(q, k, v)
        # k_cumsum = k.sum(dim=-2)
        macs += H * N * Nf
        # D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        macs += H * N * Nf
        # context = torch.einsum('...nd,...ne->...de', k, v)
        macs += H * N * Nf * C
        # out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        macs += 2 * H * N * Nf * C
        # print('macs fast att', macs / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class FastRandomHashAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0,
                 generalized_attention=False, kernel_fn=nn.ReLU(),
                 no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix,
                                         nb_rows=self.nb_features,
                                         nb_columns=dim_heads,
                                         scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        self.num_hashes = args.num_hashes
        self.__flops__ = 0

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, qk, v):
        device = qk.device

        create_kernel = partial(softmax_kernel_share_qk,
                                projection_matrix=self.projection_matrix,
                                device=device)
        qk = create_kernel(qk)

        # hashing attention
        B, H, N, D = qk.shape
        rotaion_shape = (H, D, self.num_hashes, 1)
        random_rotations = torch.randn(rotaion_shape, dtype=qk.dtype, device=qk.device)[None, ...].expand((B, H, D, self.num_hashes, 1)).reshape(B*H, D, self.num_hashes, 1)

        qk = qk.reshape(B*H, N, D)

        rotated_qk = torch.einsum('btf,bfhi->bhti', qk, random_rotations).squeeze(dim=-1).permute(0, 2, 1)
        binary_input_no_grad = torch.sign(rotated_qk)
        cliped_input = torch.clamp(rotated_qk, -1.0, 1.0)
        qk = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        qk = qk.reshape(B, H, N, self.num_hashes)

        out = linear_attention_bias(qk, qk, v)
        return out

    @staticmethod
    def compute_macs(module, input, output):
        input = input[0]
        _, H, N, C = input.shape
        Nf = module.nb_features
        assert C == module.dim_heads
        macs = 0
        n_params = 0

        if module.no_projection:
            raise ValueError("Not supported yet!")
        elif module.generalized_attention:
            raise ValueError("Not supported yet!")
        else:
            n_params += C * Nf
            # q = create_kernel(q, is_query=True)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf
            # k = create_kernel(k, is_query=False)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf

        # out = linear_attention(q, k, v)
        # k_cumsum = k.sum(dim=-2)
        macs += H * N * Nf
        # D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        macs += H * N * Nf
        # context = torch.einsum('...nd,...ne->...de', k, v)
        macs += H * N * Nf * C
        # out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        macs += 2 * H * N * Nf * C
        # print('macs fast att', macs / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class FastQuantAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0,
                 generalized_attention=False, kernel_fn=nn.ReLU(),
                 no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix,
                                         nb_rows=self.nb_features,
                                         nb_columns=dim_heads,
                                         scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        # self.num_hashes = args.num_hashes
        self.__flops__ = 0

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, qk, v):
        device = qk.device

        create_kernel = partial(softmax_kernel_share_qk,
                                projection_matrix=self.projection_matrix,
                                device=device)
        qk = create_kernel(qk)


        # quantize qk
        binary_input_no_grad = torch.sign(qk - qk.mean())
        cliped_input = torch.clamp(qk, -1.0, 1.0)
        qk = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        # # hashing attention
        # B, H, N, D = qk.shape
        # rotaion_shape = (H, D, self.num_hashes, 1)
        # random_rotations = torch.randn(rotaion_shape, dtype=qk.dtype, device=qk.device)[None, ...].expand((B, H, D, self.num_hashes, 1)).reshape(B*H, D, self.num_hashes, 1)
        #
        # qk = qk.reshape(B*H, N, D)
        #
        # rotated_qk = torch.einsum('btf,bfhi->bhti', qk, random_rotations).squeeze(dim=-1).permute(0, 2, 1)
        # binary_input_no_grad = torch.sign(rotated_qk)
        # cliped_input = torch.clamp(rotated_qk, -1.0, 1.0)
        # qk = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        # qk = qk.reshape(B, H, N, self.num_hashes)

        out = linear_attention_bias_quant(qk, qk, v)
        return out

    @staticmethod
    def compute_macs(module, input, output):
        input = input[0]
        _, H, N, C = input.shape
        Nf = module.nb_features
        assert C == module.dim_heads
        macs = 0
        n_params = 0

        if module.no_projection:
            raise ValueError("Not supported yet!")
        elif module.generalized_attention:
            raise ValueError("Not supported yet!")
        else:
            n_params += C * Nf
            # q = create_kernel(q, is_query=True)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf
            # k = create_kernel(k, is_query=False)
            macs += H * N * Nf * C + 2 * H * N * C + 2 * H * N * Nf

        # out = linear_attention(q, k, v)
        # k_cumsum = k.sum(dim=-2)
        macs += H * N * Nf
        # D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        macs += H * N * Nf
        # context = torch.einsum('...nd,...ne->...de', k, v)
        macs += H * N * Nf * C
        # out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        macs += 2 * H * N * Nf * C
        # print('macs fast att', macs / 1e8)

        module.__flops__ += macs
        # return n_params, macs



class PerformerSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., nb_features=None,
                 generalized_attention=False,
                 kernel_fn=nn.ReLU(), no_projection=False, sr_ratio=1.0, linear=False, seq_len=196):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.dim = dim
        assert dim % num_heads == 0, 'dimension must be divisible by number of heads'
        head_dim = dim // num_heads
        nb_features = args.num_features
        self.fast_attention = FastAttention(
            head_dim, nb_features, generalized_attention=generalized_attention,
            kernel_fn=kernel_fn, no_projection=no_projection
        )

        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5  # not used in performer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        x = self.fast_attention(q, k, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RePerformerRandomHash(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., nb_features=None,
                 generalized_attention=False,
                 kernel_fn=nn.ReLU(), no_projection=False, sr_ratio=1.0, linear=False, seq_len=196):
        super().__init__()
        self.dim = dim
        self.sr_ratio = sr_ratio
        assert dim % num_heads == 0, 'dimension must be divisible by number of heads'
        head_dim = dim // num_heads
        nb_features = args.num_features
        self.fast_attention = FastRandomHashAttention(
            head_dim, nb_features, generalized_attention=generalized_attention,
            kernel_fn=kernel_fn, no_projection=no_projection
        )
        self.num_heads = num_heads
        self.to_qk = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape

        qk = self.to_qk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = self.fast_attention(qk, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerformerQuantAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., nb_features=None,
                 generalized_attention=False,
                 kernel_fn=nn.ReLU(), no_projection=False, sr_ratio=1.0, linear=False, seq_len=196):
        super().__init__()
        self.dim = dim
        self.sr_ratio = sr_ratio
        assert dim % num_heads == 0, 'dimension must be divisible by number of heads'
        head_dim = dim // num_heads
        nb_features = args.num_features
        self.fast_attention = FastQuantAttention(
            head_dim, nb_features, generalized_attention=generalized_attention,
            kernel_fn=kernel_fn, no_projection=no_projection
        )
        self.num_heads = num_heads
        self.to_qk = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape

        qk = self.to_qk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = self.fast_attention(qk, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
