import torch
import numpy as np
import math
import deepshift.ste as ste
import deepshift.kernels

def round_to_fixed(input, integer_bits=16, fraction_bits=16): 
    assert integer_bits >= 1, integer_bits
    # TODO: Deal with unsigned tensors where there is no sign bit
    #       which is the case with activations to convolution that 
    #       are usually the output of a Relu layer
    if integer_bits == 1: 
        return torch.sign(input) - 1 
    delta = math.pow(2.0, -(fraction_bits))
    bound = math.pow(2.0, integer_bits-1) 
    min_val = - bound 
    max_val = bound - 1 
    rounded = torch.floor(input / delta) * delta

    clipped_value = torch.clamp(rounded, min_val, max_val)
    return clipped_value 

def get_shift_and_sign(x, rounding='deterministic'):
    sign = torch.sign(x)
    
    x_abs = torch.abs(x)
    shift = round(torch.log(x_abs) / np.log(2), rounding)
    
    return shift, sign   

# FIXME:
def get_shift_and_sign_SP2(x, weight_bits):
    sign_1 = torch.sign(x)
    sign_2 = torch.sign(x)
    x_abs = torch.abs(x)
    # TODO:
    # weight_bits_1 = (weight_bits//2)
    weight_bits_1 = 2
    weight_bits_2 = weight_bits - weight_bits_1 
    shift_1 = (torch.log(x_abs) / np.log(2)).floor()
    sign_1[shift_1 < (2**(weight_bits_1)-1)] = 0
    shift_1 = shift_1.clamp(-1 * (2**(weight_bits_1)-1), -1)
    x_diff = x_abs - torch.abs(sign_1*2**shift_1)
    shift_2 = (torch.log(x_diff) / np.log(2)).round()
    shift_2 = shift_2.clamp(-1 * (2**(weight_bits_2)-1), -1)

    delta = x_abs - sign_1*2**shift_1 - 2**shift_2
    return shift_1, shift_2, sign_1, sign_2    


def round_power_of_2(x, rounding='deterministic'):
    shift, sign = get_shift_and_sign(x, rounding)    
    x_rounded = (2.0 ** shift) * sign
    return x_rounded

def round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()

def clampabs(input, min, max):
    assert(min >= 0 and max >=0)

    input[input > max] = max
    input[input < -max] = -max

    input[(input > torch.zeros_like(input)) & (input < min)] = min
    input[(input < torch.zeros_like(input)) & (input > -min)] = -min
    return input 

class ConcWeight():
    def __init__(self, data=None, base=0, bits=8):
        self.data = data 
        self.base = base
        self.bits = bits

##concatenate shift and sign together
def compress_bits(shift, sign):
    conc_weight = ConcWeight() 

    if len(shift.shape) == 2:
        shift = shift.unsqueeze(-1).unsqueeze(-1)

    # if sign is ternary, then use a big shift value that is equivalent to multiplying by zero
    zero_sign_indices = (sign == 0).nonzero()
    shift[zero_sign_indices] = -32
    sign[zero_sign_indices] = +1

    conc_weight.bits = math.ceil(torch.log( - torch.min(shift) + 1)/ np.log(2))
    # treat shift to the right as the default
    shift = shift * -1
    minimum = int(torch.min(shift))
    if minimum < 0:
        conc_weight.base = minimum
        shift = shift - minimum
    else:
        conc_weight.base = 0

    num = int(32 / (conc_weight.bits + 1))
    row_length = int((shift.shape[1] * shift.shape[2] * shift.shape[3] + num -1) / num )
    size = row_length * shift.shape[0]

    conc_weight.data = deepshift.kernels.compress_sign_and_shift(shift.int().cuda(), sign.int().cuda(), size, conc_weight.base, conc_weight.bits, row_length, num)

    return conc_weight


def get_weight_from_ps(shift, sign, weight_bits, rounding):
    shift_range = (-1 * (2**(weight_bits - 1) - 2), 0)
    shift.data = ste.clamp(shift.data, *shift_range)
    shift_rounded = ste.round(shift, rounding=rounding)
    sign_rounded_signed = ste.sign(ste.round(sign, rounding=rounding))
    weight_ps = ste.unsym_grad_mul(2**shift_rounded, sign_rounded_signed)

    return weight_ps

def get_bias_from_shift(bias, act_integer_bits, act_fraction_bits):
    bias_fixed_point = ste.round_fixed_point(bias, act_integer_bits, act_fraction_bits)
    return bias_fixed_point

# -------------------- SP2 --------------------
def build_power_value(B=4, additive=True):
    base_a = [0.]
    base_b = [0.]
    
    if B == 4:
        for i in range(3):
            base_a.append(2 ** (-2 * i - 1))
            base_b.append(2 ** (-2 * (i-1) - 2))
        
    p1 = []
    p2 = []
    s1 = []
    s2 = []
    values = []
    for i, a in enumerate(base_a):
        for j, b in enumerate(base_b):
            values.append((a + b))
            if a == 0.:
                s2.append(0)
            else:
                s2.append(1)
            if b == 0.:
                s1.append(0)
            else:
                s1.append(1)
            if j > 0:
                p1.append(-(j-1))
            else:
                p1.append(0)
            if i > 0:
                p2.append(-(i-1))
            else: 
                p2.append(0)
    values = torch.Tensor(((values)))
    p1 = torch.Tensor(((p1)))
    p2 = torch.Tensor(((p2)))
    s1 = torch.Tensor(((s1)))
    s2 = torch.Tensor(((s2)))
    values = values.mul(1.0 / torch.max(values))
    
    return values, s1, p1, s2, p2


def get_param_APoT(x):
    value_s, s1, p1, s2, p2 = build_power_value()
    shape = x.shape
    xhard = x.view(-1)
    sign = x.sign()

    value_s = value_s.type_as(x)
    s1 = s1.type_as(x)
    s2 = s2.type_as(x)
    p1 = p1.type_as(x)
    p2 = p2.type_as(x)
    # diff = value_s - (s1*2**(2*p1) + s2*2**(2*p2-1))*2/3
    xhard = xhard.abs()
    idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
    sign_1 = s1[idxs].view(shape)
    sign_2 = s2[idxs].view(shape)
    shift_1 = p1[idxs].view(shape)
    shift_2 = p2[idxs].view(shape)
    # # TODO:
    # weights = sign*(sign_1*2**(2*shift_1)+sign_2*2**(2*shift_2-1))*2/3
    # diff = x - weights
    return sign, sign_1, sign_2, shift_1, shift_2
    

