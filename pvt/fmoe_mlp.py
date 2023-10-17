r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
# from fmoe.layers import FMoE
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from deepshift.modules import LinearShift, Conv2dShift
# from fmoe.gates import NaiveGate, NoisyGate
import tree
from fmoe_new import SparseDispatcher, NaiveGate, NoisyGate

act_integer_bits=16 
act_fraction_bits=16
weight_bits=5

# from fmoe_new import SparseDispatcher, NaiveGate

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
        # self.dwconv = DWConv(hidden_features)
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

    def forward(self, x,):
        self.shape = []
        # self.H = H
        # self.W = W
        self.shape.append(x.shape)
        x = self.fc1(x)
        # print(torch.isnan(x).any())
        if self.linear:
            x = self.relu(x)
        self.shape.append(x.shape)
        # x = self.dwconv(x, H, W)
        # print(torch.isnan(x).any())
        x = self.act(x)
        # print('ok')
        x = self.drop(x)
        self.shape.append(x.shape)
        x = self.fc2(x)
        # print(torch.isnan(x).any())
        x = self.drop(x)
            
        return x


def Shift_Linear(in_features, out_features, convert_weights=True, freeze_sign=False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits, SP2=False, bias=True):
    
    return LinearShift(in_features, out_features, bias, freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits, SP2=SP2)


def Shift_Conv(in_channels, out_channels, kernel_size, stride, padding, bias, groups, 
               convert_weights=True, freeze_sign=False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits, SP2=False, padding_mode='zeros', dilation=1):
    
    return Conv2dShift(in_channels, out_channels, kernel_size, stride,
                                                    padding, dilation, groups=groups,
                                                    bias=bias, padding_mode=padding_mode,
                                                    freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                    weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits,
                                                    SP2=SP2)


class Shift_DWConv(nn.Module):
    def __init__(self, dim=768):
        super(Shift_DWConv, self).__init__()
        self.dwconv = Shift_Conv(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Shift_Mlp(nn.Module):
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
        # print(hidden_features)
        self.fc1 = Shift_Linear(in_features, hidden_features)
        # self.dwconv = Shift_DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = Shift_Linear(hidden_features, out_features)
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

    def forward(self, x,):
        self.shape = []
        # self.H = H
        # self.W = W
        self.shape.append(x.shape)
        x = self.fc1(x)
        # print(torch.isnan(x).any())
        if self.linear:
            x = self.relu(x)
        self.shape.append(x.shape)
        # x = self.dwconv(x, H, W)
        # print(torch.isnan(x).any())
        x = self.act(x)
        # print('ok')
        x = self.drop(x)
        self.shape.append(x.shape)
        x = self.fc2(x)
        # print(torch.isnan(x).any())
        x = self.drop(x)
            
        return x


# class FMoETransformerMLP(FMoE):
#     r"""
#     A complete MoE MLP module in a Transformer block.
#     * `activation` is the activation function to be used in MLP in each expert.
#     * `d_hidden` is the dimension of the MLP layer.
#     """

#     def __init__(
#         self,
#         num_expert=2,
#         d_model=None,
#         d_hidden=None,
#         activation=nn.GELU,
#         expert_dp_comm="none",
#         expert_rank=0,
#         drop=0.0,
#         linear=False,
#         world_size=1,
#         top_k=1,
#         gate=NaiveGate,
#         **kwargs
#     ):
#         super().__init__(num_expert=num_expert, d_model=d_model, gate=gate, world_size=world_size, top_k=top_k, **kwargs)
#         self.experts = nn.ModuleList([
#             Mlp(in_features=d_model, hidden_features=d_hidden, act_layer=activation,
#             drop=drop, linear=linear),
#             # Mlp(in_features=d_model, hidden_features=d_hidden, act_layer=activation,
#             # drop=drop, linear=linear),
#             # QMlp(in_features=d_model, hidden_features=d_hidden, act_layer=activation,
#             # drop=drop, linear=linear),
#             Shift_Mlp(in_features=d_model, hidden_features=d_hidden, act_layer=activation,
#             drop=drop, linear=linear),
#             # Shift_Mlp(in_features=d_model, hidden_features=d_hidden//2, act_layer=activation,
#             # drop=drop, linear=linear),
#         ])
#         self.mark_parallel_comm(expert_dp_comm)

#     def forward(self, inp: torch.Tensor,H,W):
#         r"""
#         This module wraps up the FMoE module with reshape, residual and layer
#         normalization.
#         """
#         original_shape = inp.shape
#         inp = inp.reshape(-1, self.d_model)
#         output = super().forward(inp,H,W)
#         return output.reshape(original_shape)
    


class PyTorchFMoE_MLP(nn.Module):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=2,
        d_model=None,
        d_hidden=None,
        activation=nn.GELU,
        expert_dp_comm="none",
        expert_rank=0,
        drop=0.0,
        linear=False,
        world_size=1,
        top_k=1,
        gate=NaiveGate,
        **kwargs
    ):
        super(PyTorchFMoE_MLP, self).__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.top_k = top_k
        self.gate_type = gate
        self.experts = nn.ModuleList([
            Mlp(in_features=d_model, hidden_features=d_hidden, act_layer=activation,
            drop=drop, linear=linear),
        
            Shift_Mlp(in_features=d_model, hidden_features=d_hidden, act_layer=activation,
            drop=drop, linear=linear),
        ])
        

    def forward(self, inp: torch.Tensor, shift_ratio=None):
        r"""
        Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        # moe_inp_batch_size = tree.flatten(
        #     tree.map_structure(lambda tensor: tensor.shape[0], inp)
        # )
        # assert all(
        #     [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        # ), "MoE inputs must have the same batch size"
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        if self.gate_type == NaiveGate:
            # if self.gate.observe:
            #     gates, shift_ratio = self.gate(inp)
            # else:
            #     gates = self.gate(inp, shift_ratio)
            gates = self.gate(inp, shift_ratio)
        
        elif self.gate_type == NoisyGate or self.gate_type == NoisyGate:  
            gates = self.gate(inp)
            
        # inp = inp.reshape(-1, self.d_model)
        dispatcher = SparseDispatcher(self.num_expert, gates, self.top_k)
        expert_inputs = dispatcher.dispatch(inp)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_expert)]
        y = dispatcher.combine(expert_outputs)
        
        # if self.gate.observe:
        #     return y.reshape(original_shape), shift_ratio
        return y.reshape(original_shape)
        # return y