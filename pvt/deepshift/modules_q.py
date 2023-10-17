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
from pdb import set_trace

# Inherit from Function
class LinearShiftQFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, conc_weight=None, use_kernel=False, use_cuda=True, rounding='deterministic', shift_range=(0,-15), act_integer_bits=16, act_fraction_bits=16):
        shift, sign = utils.get_shift_and_sign(weight, rounding)
   
        if use_kernel:
            input_fixed_point = (input * (2 ** act_fraction_bits)).int()
            if bias is not None:
                bias_fixed_point = (bias * (2 ** act_fraction_bits)).int()

            out = deepshift.kernels.linear(input_fixed_point, shift, sign, bias_fixed_point, conc_weight, use_cuda)
            out = out.float()
            out = out / (2**act_fraction_bits)
        else:
            input.data = utils.round_to_fixed(input.data, act_integer_bits, act_fraction_bits)
            if bias is not None:
                bias.data = utils.round_to_fixed(bias.data, act_integer_bits, act_fraction_bits)

            weight_s = (2.0 ** shift) * sign
            out = input.mm(weight_s.t())
            if bias is not None:
                out += bias.unsqueeze(0).expand_as(out)

            ctx.save_for_backward(input, weight_s, bias)

        return out

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight_s, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_s)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input) # * v * math.log(2)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None


def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values

def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y

def apot_quantization(tensor, alpha, proj_set, is_weight=True, grad_scale=None):
    def power_quant(x, value_s):
        if is_weight:
            shape = x.shape
            xhard = x.view(-1)
            sign = x.sign()
            # set_trace()
            value_s = value_s.type_as(x)
            xhard = xhard.abs()
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape).mul(sign)
            xhard = xhard
        else:
            shape = x.shape
            xhard = x.view(-1)
            value_s = value_s.type_as(x)
            xhard = xhard
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape)
            xhard = xhard
        xout = (xhard - x).detach() + x
        return xout

    if grad_scale:
        alpha = gradient_scale(alpha, grad_scale)
    data = tensor / alpha
    # data = tensor 
    if is_weight:
        # TODO:
        data = data.clamp(-1, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha
    else:
        data = data.clamp(0, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha
    return data_q


class LinearShiftQ(nn.Module):
    def __init__(self, in_features, out_features, bias=True, check_grad=False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=5, act_integer_bits=16, act_fraction_bits=16, SP2=False):
 
        super(LinearShiftQ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_kernel = use_kernel
        self.check_grad = check_grad
        self.use_cuda = use_cuda
        self.conc_weight = None
        self.rounding = rounding
        self.shift_range = (-1 * (2**(weight_bits - 1) - 1), 0) # we use binary weights to represent sign
        self.act_integer_bits, self.act_fraction_bits = act_integer_bits, act_fraction_bits
        self.SP2 = SP2
        if self.SP2:
            print("Use APoT!!!")
            self.proj_set_weight = build_power_value(B=weight_bits-1, additive=True)
            self.weight_alpha = torch.nn.Parameter(torch.tensor(1.0))
            # self.weight_alpha = None

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

        self.weight = nn.Parameter(tensor_constructor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.SP2:
            # ########## weight normalization ##########
            # mean = self.weight.mean()
            # std = self.weight.std()
            # weight = self.weight.add(-mean).div(std)
            weight = self.weight
            weight_q = apot_quantization(weight, self.weight_alpha, self.proj_set_weight, is_weight=True, grad_scale=None)
        else:
            self.weight.data = ste.clampabs(self.weight.data, 2**self.shift_range[0], 2**self.shift_range[1])
            weight_q = ste.round_power_of_2(self.weight, self.rounding)
        # loss = weight_q - self.weight
        input_fixed_point = ste.round_fixed_point(input, self.act_integer_bits, self.act_fraction_bits)
        # FIXME:
        # loss = input_fixed_point - input
        # input_fixed_point = input
        if self.bias is not None:
            bias_fixed_point = ste.round_fixed_point(self.bias, self.act_integer_bits, self.act_fraction_bits)
            # bias_fixed_point = self.bias
        else:
            bias_fixed_point = None
            
        if self.use_kernel:
            return LinearShiftQFunction.apply(input_fixed_point, weight_q, bias_fixed_point, self.conc_weight, self.use_kernel, self.use_cuda, self.act_integer_bits, self.act_fraction_bits)
        else:
            # FIXME:
            # out = input_fixed_point.mm(weight_q.t())
            # if self.bias is not None:
            #     out += self.bias.unsqueeze(0).expand_as(out)

            out = F.linear(input_fixed_point, weight_q, bias_fixed_point)   
            # out = F.linear(input_fixed_point, weight_q)  
            # if self.bias is not None:
            #     out += self.bias.unsqueeze(0).expand_as(out)

            return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# Inherit from Function
class Conv2dShiftQFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, conc_weight=None, stride=1, padding=0, dilation=1, groups=1, use_kernel=False, use_cuda=False, act_integer_bits=16, act_fraction_bits=16):
        shift, sign = utils.get_shift_and_sign(weight, rounding='deterministic')

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
            weight_s = (2.0 ** shift) * sign
            out = F.conv2d(input, weight_s, bias, stride, padding, dilation, groups)

            ctx.save_for_backward(input, weight_s, bias)
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
        input, weight_s, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight_s, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight_s.shape, grad_output, stride, padding, dilation, groups) # * v * math.log(2)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class _ConvNdShiftQ(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, 
                 check_grad=False,
                 rounding='deterministic', 
                 weight_bits=5, act_integer_bits=16, act_fraction_bits=16):
        super(_ConvNdShiftQ, self).__init__()
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
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.rounding = rounding
        self.shift_range = (-1 * (2**(weight_bits - 1) - 1), 0) # we use binary weights to represent sign
        self.act_integer_bits, self.act_fraction_bits = act_integer_bits, act_fraction_bits
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
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

class Conv2dShiftQ(_ConvNdShiftQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', 
                 check_grad=False, use_kernel=False, use_cuda =True,
                 rounding='deterministic', 
                 weight_bits=5, act_integer_bits=16, act_fraction_bits=16):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.use_kernel = use_kernel
        self.use_cuda = use_cuda
        self.conc_weight = None
        super(Conv2dShiftQ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode,
            check_grad, rounding, 
            weight_bits, act_integer_bits, act_fraction_bits)

    #@weak_script_method
    def forward(self, input):
        self.weight.data = ste.clampabs(self.weight.data, 2**self.shift_range[0], 2**self.shift_range[1])     
        weight_q = ste.round_power_of_2(self.weight, self.rounding)
        # FIXME:
        input_fixed_point = ste.round_fixed_point(input, self.act_integer_bits, self.act_fraction_bits)
        # input_fixed_point = input
        if self.bias is not None:
            bias_fixed_point = ste.round_fixed_point(self.bias, self.act_integer_bits, self.act_fraction_bits)
            # bias_fixed_point = self.bias
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
            return Conv2dShiftQFunction.apply(input_padded, weight_q, bias_fixed_point, self.conc_weight, 
                                              self.stride, padding, self.dilation, self.groups, 
                                              self.use_kernel, self.use_cuda,
                                              self.act_integer_bits, self.act_fraction_bits)
        else:
            return torch.nn.functional.conv2d(input_padded, weight_q, bias_fixed_point, 
                                              self.stride, padding, self.dilation, self.groups)
