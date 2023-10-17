import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
import numpy as np
import time

import deepshift.utils as utils
import deepshift.kernels
import deepshift.ste as ste

log2 = math.log(2)

# Inherit from Function
class LinearShiftFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, shift, sign, bias=None, conc_weight=None, use_kernel=False, use_cuda=True, rounding='deterministic', shift_range=(-14, 0), act_integer_bits=16, act_fraction_bits=16):
        if use_kernel:
            input_fixed_point = (input * (2 ** act_fraction_bits)).int()
            if bias is not None:
                bias_fixed_point = (bias * (2 ** act_fraction_bits)).int()

            out = deepshift.kernels.linear(input_fixed_point, shift, sign, bias_fixed_point, conc_weight, use_cuda)
            out = out.float()
            out = out / (2**act_fraction_bits)
        else:
            sign = sign.clamp(-1,1)
            shift = shift.clamp(*shift_range)
            input.data = utils.round_to_fixed(input.data, act_integer_bits, act_fraction_bits)
            # FIXME: don't modify bias.data, but instead create a new tensor?
            if bias is not None:
                bias.data = utils.round_to_fixed(bias.data, act_integer_bits, act_fraction_bits)

            v = 2**shift.round() * sign.round().sign()
            out = input.mm(v.t())
            if bias is not None:
                out += bias.unsqueeze(0).expand_as(out)
            
            ctx.save_for_backward(input, shift, sign, bias, v)

        return out

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, shift, sign, bias, v = ctx.saved_tensors
        grad_input = grad_shift = grad_sign = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(v)
        if ctx.needs_input_grad[2]:
            grad_sign = grad_output.t().mm(input)
        if ctx.needs_input_grad[1]:
            if grad_sign is None:
                grad_shift = grad_output.t().mm(input) * v * log2
            else:
                grad_shift = grad_sign * v * log2
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_shift, grad_sign, grad_bias, None, None, None


class LinearShift(nn.Module):
    def __init__(self, in_features, out_features, bias=True, check_grad=False, freeze_sign=False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=5, act_integer_bits=16, act_fraction_bits=16, SP2=False):
        super(LinearShift, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_kernel = use_kernel
        self.check_grad = check_grad
        self.use_cuda = use_cuda
        self.conc_weight = None
        self.rounding = rounding
        self.SP2 = SP2
        # TODO:
        # self.shift_range = (-1 * (2**(weight_bits - 1) - 2), 0) # we use ternary weights to represent sign
        if self.SP2:
            weight_bits_sp2 = (weight_bits-1)//2
            print("Use APoT!!!")
            # weight_bits_1 = 2
            # FIXME: weight_bits = 5bit
            self.shift_range = (-2, 0)
            self.sign_range = (0, 1)
            # self.shift_range_2 = (-1 * (2**(weight_bits_sp2)-1), 0)
            # self.proj_set_weight = build_power_value(B=weight_bits, additive=True)
        else:
            self.shift_range = (-1 * (2**(weight_bits - 1) - 2), 0) # we use ternary weights to represent sign
        self.act_integer_bits, self.act_fraction_bits = act_integer_bits, act_fraction_bits
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        
        if check_grad:
            tensor_constructor = torch.DoubleTensor # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor # In PyTorch torch.Tensor is alias torch.FloatTensor
        if self.SP2:
            self.shift_1 = nn.Parameter(tensor_constructor(out_features, in_features))
            self.shift_2 = nn.Parameter(tensor_constructor(out_features, in_features))
            self.sign_1 = nn.Parameter(tensor_constructor(out_features, in_features), requires_grad = (freeze_sign == False))
            self.sign_2 = nn.Parameter(tensor_constructor(out_features, in_features), requires_grad = (freeze_sign == False))
            self.sign = nn.Parameter(tensor_constructor(out_features, in_features), requires_grad = (freeze_sign == False))
            # self.weight_alpha = torch.nn.Parameter(torch.tensor(1.0))
        else:
            # TODO:
            self.shift = nn.Parameter(tensor_constructor(out_features, in_features))
            self.sign = nn.Parameter(tensor_constructor(out_features, in_features), requires_grad = (freeze_sign == False))
            # self.shift = nn.Parameter(tensor_constructor(out_features, in_features), requires_grad=False)
            # self.sign = nn.Parameter(tensor_constructor(out_features, in_features), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.SP2:
            self.shift_1.data.uniform_(*self.shift_range) 
            self.shift_2.data.uniform_(*self.shift_range) 
            self.sign.data.uniform_(-1, 1) 
            self.sign_1.data.uniform_(0, 1) 
            self.sign_2.data.uniform_(0, 1) 
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift_1)
            # pass
        else:
            self.shift.data.uniform_(*self.shift_range) 
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift)
            self.sign.data.uniform_(-1, 1) 
        
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.SP2:
            self.shift_1.data = ste.clamp(self.shift_1.data, *self.shift_range)
            self.shift_2.data = ste.clamp(self.shift_2.data, *self.shift_range)
            shift_rounded_1 = ste.round(self.shift_1, rounding=self.rounding)
            shift_rounded_2 = ste.round(self.shift_2, rounding=self.rounding)
            self.sign_1.data = ste.clamp(self.sign_1.data, *self.sign_range)
            self.sign_2.data = ste.clamp(self.sign_2.data, *self.sign_range)
            sign_1_rounded_signed = ste.sign(ste.round(self.sign_1, rounding=self.rounding))
            sign_2_rounded_signed = ste.sign(ste.round(self.sign_2, rounding=self.rounding))
            sign_rounded_signed = ste.sign(ste.round(self.sign, rounding=self.rounding))
            # sign_rounded_signed_2 = ste.sign(ste.round(self.sign_2, rounding=self.rounding))
            weight_ps_1 = ste.unsym_grad_mul(2**(2*shift_rounded_1), sign_1_rounded_signed)
            weight_ps_2 = ste.unsym_grad_mul(2**(2*shift_rounded_2-1), sign_2_rounded_signed)
            weight_ps = ste.unsym_grad_mul((weight_ps_1 + weight_ps_2)*2/3, sign_rounded_signed)
    
        else:
            self.shift.data = ste.clamp(self.shift.data, *self.shift_range)
            shift_rounded = ste.round(self.shift, rounding=self.rounding)
            sign_rounded_signed = ste.sign(ste.round(self.sign, rounding=self.rounding))
            weight_ps = ste.unsym_grad_mul(2**shift_rounded, sign_rounded_signed)

        # TODO: round bias and input to fixed point
        input_fixed_point = ste.round_fixed_point(input, self.act_integer_bits, self.act_fraction_bits)
        if self.bias is not None:
            bias_fixed_point = ste.round_fixed_point(self.bias, self.act_integer_bits, self.act_fraction_bits)
        else:
            bias_fixed_point = None

        if self.use_kernel:
            return LinearShiftFunction.apply(input_fixed_point, self.shift, self.sign, bias_fixed_point, self.conc_weight, self.use_kernel, self.use_cuda, self.rounding, self.shift_range, self.act_integer_bits, self.act_fraction_bits)
        else:
            # return torch.nn.functional.linear(input, weight_ps, self.bias)
            return torch.nn.functional.linear(input_fixed_point, weight_ps, bias_fixed_point)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# Inherit from Function
class Conv2dShiftFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, shift, sign, bias=None, conc_weight=None, stride=1, padding=0, dilation=1, groups=1, use_kernel=False, use_cuda=False, rounding='deterministic', shift_range=(-14,0), act_integer_bits=16, act_fraction_bits=16):
        # start_time = time.time()
        if use_kernel:
            input_fixed_point = (input * (2 ** act_fraction_bits)).int()
            if bias is not None:
                bias_fixed_point = (bias * (2 ** act_fraction_bits)).int()
            else:
                bias_fixed_point = None

            out = deepshift.kernels.conv2d(input_fixed_point, shift, sign, bias_fixed_point, conc_weight, stride, padding, dilation, groups, use_cuda)

            out = out.float()
            out = out / (2**act_fraction_bits)               
        else:
            shift = shift.clamp(*shift_range)
            sign = sign.clamp(-1,1)
            input.data = utils.round_to_fixed(input.data, act_integer_bits, act_fraction_bits)

            if bias is not None:
                bias.data = utils.round_to_fixed(bias.data, act_integer_bits, act_fraction_bits)

            shift_rounded = utils.round(shift, stochastic=False)
            sign_rounded_signed = torch.sign(utils.round(sign, stochastic=False))
            v = 2**shift_rounded * sign_rounded_signed
            out = F.conv2d(input, v, bias, stride, padding, dilation, groups)

            ctx.save_for_backward(input, shift, sign, bias, v)
            ctx.stride = stride
            ctx.padding = padding 
            ctx.dilation = dilation
            ctx.groups = groups

        return out

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, shift, sign, bias, v = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_shift = grad_sign = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, v, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[2]:
            grad_sign = torch.nn.grad.conv2d_weight(input, v.shape, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            if grad_sign is None:
                grad_shift = torch.nn.grad.conv2d_weight(input, v.shape, grad_output, stride, padding, dilation, groups) * v * log2
            else:
                grad_shift = grad_sign * v * log2
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        return grad_input, grad_shift, grad_sign, grad_bias, None, None, None, None, None, None, None, None

class _ConvNdShift(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, 
                 check_grad=False, freeze_sign=False, rounding='deterministic', 
                 weight_bits=5, act_integer_bits=16, act_fraction_bits=16, SP2=False):
        super(_ConvNdShift, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.rounding=rounding
        self.SP2 = SP2
        # TODO:
        # self.shift_range = (-1 * (2**(weight_bits - 1) - 2), 0) # we use ternary weights to represent sign
        if self.SP2:
            # weight_bits_1 = (weight_bits//2)
            weight_bits_1 = 2
            weight_bits_2 = weight_bits - weight_bits_1 
            self.shift_range_1 = (-1 * (2**(weight_bits_1)-1), -1)
            self.shift_range_2 = (-1 * (2**(weight_bits_2)-1), -1)
        else:
            self.shift_range = (-1 * (2**(weight_bits - 1) - 2), 0) # we use ternary weights to represent sign
        self.act_integer_bits, self.act_fraction_bits = act_integer_bits, act_fraction_bits

        if check_grad:
            tensor_constructor = torch.DoubleTensor # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor # In PyTorch torch.Tensor is alias torch.FloatTensor

        if transposed:
            if self.SP2:
                self.shift_1 = nn.Parameter(tensor_constructor(out_channels, in_channels // groups, *kernel_size))
                self.shift_2 = nn.Parameter(tensor_constructor(out_channels, in_channels // groups, *kernel_size))
                self.sign_1 = nn.Parameter(tensor_constructor(
                    in_channels, out_channels // groups, *kernel_size), 
                    requires_grad = (freeze_sign == False))
                self.sign_2 = nn.Parameter(tensor_constructor(
                    in_channels, out_channels // groups, *kernel_size), 
                    requires_grad = (freeze_sign == False))
            else:
                self.shift = nn.Parameter(tensor_constructor(out_channels, in_channels // groups, *kernel_size))
                self.sign = nn.Parameter(tensor_constructor(
                    in_channels, out_channels // groups, *kernel_size), 
                    requires_grad = (freeze_sign == False))
        else:
            if self.SP2:
                self.shift_1 = nn.Parameter(tensor_constructor(out_channels, in_channels // groups, *kernel_size))
                self.shift_2 = nn.Parameter(tensor_constructor(out_channels, in_channels // groups, *kernel_size))
                self.sign_1 = nn.Parameter(tensor_constructor(
                    in_channels, out_channels // groups, *kernel_size), 
                    requires_grad = (freeze_sign == False))
                self.sign_2 = nn.Parameter(tensor_constructor(
                    in_channels, out_channels // groups, *kernel_size), 
                    requires_grad = (freeze_sign == False))
            else:
                self.shift = nn.Parameter(tensor_constructor(out_channels, in_channels // groups, *kernel_size))
                self.sign = nn.Parameter(tensor_constructor(
                    out_channels, in_channels // groups, *kernel_size),
                    requires_grad = (freeze_sign == False))
        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.SP2:
            self.shift_1.data.uniform_(-10, -1) # (-0.1, 0.1)
            self.shift_2.data.uniform_(-10, -1) # (-0.1, 0.1)
            self.sign_1.data.uniform_(-1, 1) # (-0.1, 0.1)
            self.sign_2.data.uniform_(-1, 1) # (-0.1, 0.1)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift_1)
        else:
            self.shift.data.uniform_(-10, -1) # (-0.1, 0.1)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift)
            self.sign.data.uniform_(-1, 1) # (-0.1, 0.1)
        
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dShift(_ConvNdShift):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', 
                 check_grad=False, freeze_sign=False, use_kernel=False, use_cuda=True, rounding='determinstic', 
                 weight_bits=5, act_integer_bits=16, act_fraction_bits=16, SP2=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.use_kernel = use_kernel
        self.use_cuda = use_cuda
        self.conc_weight = None
        super(Conv2dShift, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, 
            check_grad, freeze_sign, rounding,
            weight_bits, act_integer_bits, act_fraction_bits, SP2)

    #@weak_script_method
    def forward(self, input):
        # self.shift.data = ste.clamp(self.shift.data, *self.shift_range)
        # shift_rounded = ste.round(self.shift, self.rounding)
        # sign_rounded_signed = ste.sign(ste.round(self.sign, self.rounding))
        # weight_ps = ste.unsym_grad_mul(2**shift_rounded, sign_rounded_signed)
        if self.SP2:
            self.shift_1.data = ste.clamp(self.shift_1.data, *self.shift_range_1)
            self.shift_2.data = ste.clamp(self.shift_2.data, *self.shift_range_2)
            shift_rounded_1 = ste.round(self.shift_1, rounding=self.rounding)
            shift_rounded_2 = ste.round(self.shift_2, rounding=self.rounding)
            sign_rounded_signed_1 = ste.sign(ste.round(self.sign_1, rounding=self.rounding))
            sign_rounded_signed_2 = ste.sign(ste.round(self.sign_2, rounding=self.rounding))
            weight_ps_1 = ste.unsym_grad_mul(2**shift_rounded_1, sign_rounded_signed_1)
            weight_ps_2 = ste.unsym_grad_mul(2**shift_rounded_2, sign_rounded_signed_2)
            weight_ps = weight_ps_1 + weight_ps_2
        else:
            self.shift.data = ste.clamp(self.shift.data, *self.shift_range)
            shift_rounded = ste.round(self.shift, rounding=self.rounding)
            sign_rounded_signed = ste.sign(ste.round(self.sign, rounding=self.rounding))
            weight_ps = ste.unsym_grad_mul(2**shift_rounded, sign_rounded_signed)
        input_fixed_point = ste.round_fixed_point(input, self.act_integer_bits, self.act_fraction_bits)
        if self.bias is not None:
            bias_fixed_point = ste.round_fixed_point(self.bias, self.act_integer_bits, self.act_fraction_bits)
        else:
            bias_fixed_point = None

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            input_padded = F.pad(input_fixed_point, expanded_padding, mode='circular')
            padding =  _pair(0)
        else:
            input_padded = input_fixed_point
            padding = self.padding

        if self.use_kernel:
            return Conv2dShiftFunction.apply(input_padded, self.shift, self.sign, bias_fixed_point, self.conc_weight, 
                                              self.stride, padding, self.dilation, self.groups, 
                                              self.use_kernel, self.use_cuda, self.rounding, 
                                              self.shift_range, self.act_integer_bits, self.act_fraction_bits)
        else:
            return torch.nn.functional.conv2d(input_padded, weight_ps, bias_fixed_point, 
                                              self.stride, padding, self.dilation, self.groups)
