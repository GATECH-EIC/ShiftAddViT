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
# from quantize_hm.quantize import QLinear
from fmoe_new import SparseDispatcher, NaiveGate, NoisyGate, NoisyGate_V2

act_integer_bits=16 
act_fraction_bits=16
weight_bits=5


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
    

class Mlp_FMoE(nn.Module):
    def __init__(
        self,
        d_model,
        d_hidden=None,
        out_features=None,
        activation=nn.GELU,
        drop=0.0,
        linear=False,
        world_size=1,
        top_k=1,
        gate=NoisyGate
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.out_features = out_features
        self.activation = activation
        self.drop = drop
        self.linear = linear

        out_features = out_features or d_model
        d_hidden = d_hidden or d_model

        self.fc1 = PyTorchFMoE_FC(d_model, d_hidden, gate=gate)
        self.dwconv = DWConv(d_hidden)
        self.act = activation()
        self.fc2 = PyTorchFMoE_FC(d_hidden, out_features, gate=gate)
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



# class FMoETransformerFC(FMoE):
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
#         expert_dp_comm="none",
#         expert_rank=0,
#         world_size=1,
#         top_k=1,
#         gate=NaiveGate,
#         **kwargs
#     ):
#         self.d_hidden = d_hidden
#         super().__init__(num_expert=num_expert, d_model=d_model, gate=gate, world_size=world_size, top_k=top_k, **kwargs)
#         self.experts = nn.ModuleList([
#             nn.Linear(d_model, d_hidden),
#             Shift_Linear(d_model, d_hidden)
#         ])
#         self.mark_parallel_comm(expert_dp_comm)

#     def forward(self, inp: torch.Tensor):
#         r"""
#         This module wraps up the FMoE module with reshape, residual and layer
#         normalization.
#         """
#         B, N, C = inp.shape
#         # original_shape = inp.shape
#         inp = inp.reshape(-1, self.d_model)
#         output = super().forward(inp)
#         return output.reshape(B, N, self.d_hidden)
    

class PyTorchFMoE_FC(nn.Module):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        d_model=None,
        d_hidden=None,
        num_expert=2,
        bias=True,
        world_size=1,
        top_k=1,
        # TODO:
        gate=NaiveGate,
        _print=False,
        **kwargs
    ):
        super(PyTorchFMoE_FC, self).__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.gate_type = gate
        self.gate = gate(d_model, num_expert, world_size, top_k, _print=_print)
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_hidden, bias=bias),
            Shift_Linear(d_model, d_hidden, bias=bias)
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
        B, N, C = inp.shape
        inp = inp.reshape(-1, self.d_model)
        if self.gate_type == NaiveGate:
            # if self.gate.observe:
            #     gates, shift_ratio = self.gate(inp)
            # else:
            #     gates = self.gate(inp, shift_ratio)
            gates = self.gate(inp, shift_ratio)
        
        elif self.gate_type == NoisyGate or self.gate_type == NoisyGate_V2:  
            gates = self.gate(inp)
            
        # inp = inp.reshape(-1, self.d_model)
        dispatcher = SparseDispatcher(self.num_expert, gates, self.top_k)
        # dispatcher_2 = SparseDispatcher(self.num_expert, gate_2, self.top_k)
        # print(torch.equal(dispatcher._batch_index  , dispatcher_2._batch_index  ))
        # print(dispatcher._part_sizes == dispatcher_2._part_sizes )
        # print(torch.equal(dispatcher._nonzero_gates, dispatcher_2._nonzero_gates))
        expert_inputs = dispatcher.dispatch(inp)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_expert)]
        y = dispatcher.combine(expert_outputs)
        
        # if self.gate_type == NaiveGate:  
        # if self.gate.observe:
        #     return y.reshape(B, N, self.d_hidden), shift_ratio
        return y.reshape(B, N, self.d_hidden)