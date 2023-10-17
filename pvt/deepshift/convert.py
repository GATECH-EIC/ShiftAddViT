import torch
import torch.nn as nn
import numpy as np
import math
import copy

import deepshift.modules
import deepshift.modules_q
import deepshift.utils as utils
from deepshift.modules import LinearShift, Conv2dShift

def convert_to_shift(model, shift_depth, shift_type, convert_all_linear=True, convert_weights=False, freeze_sign = False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=5, act_integer_bits=16, act_fraction_bits=16, linear_count=0, conv_count=0, SP2=False):
    conversion_count = 0
    # linear_count = 0
    # conv_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], num_converted, linear_count, conv_count = convert_to_shift(model=module, shift_depth=shift_depth-conversion_count, shift_type=shift_type, convert_all_linear=convert_all_linear, convert_weights=convert_weights, freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                                   weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits,
                                                                   linear_count=linear_count,
                                                                   conv_count=conv_count, SP2=SP2)
            conversion_count += num_converted
        # if conversion_count > 9:
        if type(module) == nn.Linear:
            linear = module
            # print(name)
            # linear_count > 8
            if (name == 'to_qk' or name == 'to_v' or name == 'proj') and linear_count > 0:
            # if ((name == 'to_qk' or name == 'to_v' or name == 'proj') and linear_count > 12):
                if shift_type == 'Q':
                    shift_linear = deepshift.modules_q.LinearShiftQ(module.in_features, module.out_features, module.bias is not None, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                                    weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits,
                                                                    SP2=SP2) 
                    shift_linear.weight = linear.weight
                    if linear.bias is not None:
                        # FIXME:
                        # shift_linear.bias.data = utils.round_to_fixed(linear.bias, integer_bits=act_integer_bits, fraction_bits=act_fraction_bits)
                        shift_linear.bias.data = linear.bias

                    if use_cuda==True and use_kernel == True:
                        shift_linear.conc_weight = utils.compress_bits(*utils.get_shift_and_sign(linear.weight))
                elif shift_type == 'PS':
                    shift_linear = deepshift.modules.LinearShift(module.in_features, module.out_features, module.bias is not None, freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                                weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits,
                                                                SP2=SP2)

                    if convert_weights == True:
                        if not SP2:
                            shift_linear.shift.data, shift_linear.sign.data = utils.get_shift_and_sign(linear.weight)
                        else:
                            shift_linear.sign.data, shift_linear.sign_1.data, shift_linear.sign_2.data, shift_linear.shift_1.data, shift_linear.shift_2.data = utils.get_param_APoT(linear.weight.data)
                            # pass
                        shift_linear.bias = linear.bias
                    
                        if use_cuda==True and use_kernel == True:
                            shift_linear.conc_weight = utils.compress_bits(shift_linear.shift.data, shift_linear.sign.data)
                else:
                    raise ValueError('Unsupported shift_type argument: ', shift_type)

                model._modules[name] = shift_linear
                # if convert_all_linear == False:
            linear_count += 1

        # if type(module) == nn.Conv2d and conversion_count < shift_depth:
        #     conv2d = module
        #     # print(name)
        #     # conv_count > 1
        #     # if (name != "proj" and name != "dwconv") and conv_count > 0:
        #     if (name != "proj" and name != 'dwconv'):
        #         if shift_type == 'Q':
        #             shift_conv2d = deepshift.modules_q.Conv2dShiftQ(module.in_channels, module.out_channels, module.kernel_size, module.stride,
        #                                             module.padding, module.dilation, module.groups,
        #                                             module.bias is not None, module.padding_mode, 
        #                                             use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
        #                                             weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits) 
        #             shift_conv2d.weight = conv2d.weight
        #             if conv2d.bias is not None:
        #                 # FIXME:
        #                 # shift_conv2d.bias.data = utils.round_to_fixed(conv2d.bias, integer_bits=act_integer_bits, fraction_bits=act_fraction_bits)
        #                 shift_conv2d.bias.data = conv2d.bias

        #             if use_cuda==True and use_kernel == True:
        #                 shift_conv2d.conc_weight = utils.compress_bits(*utils.get_shift_and_sign(conv2d.weight))

        #         elif shift_type == 'PS':
        #             shift_conv2d = deepshift.modules.Conv2dShift(module.in_channels, module.out_channels, module.kernel_size, module.stride,
        #                                             module.padding, module.dilation, module.groups,
        #                                             module.bias is not None, module.padding_mode,
        #                                             freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
        #                                             weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits,
        #                                             SP2=SP2)

        #             if convert_weights == True:
        #                 # shift_conv2d.shift.data, shift_conv2d.sign.data = utils.get_shift_and_sign(conv2d.weight)
        #                 if not SP2:
        #                     shift_conv2d.shift.data, shift_conv2d.sign.data = utils.get_shift_and_sign(conv2d.weight)
        #                 else:
        #                     # shift_conv2d.shift_1.data, shift_conv2d.shift_2.data, shift_conv2d.sign_1.data, shift_conv2d.sign_2.data = utils.get_shift_and_sign_SP2(conv2d.weight, weight_bits)
        #                     pass
        #                 shift_conv2d.bias = conv2d.bias

        #             if use_cuda==True and use_kernel == True:
        #                 shift_conv2d.conc_weight = utils.compress_bits(shift_conv2d.shift.data, shift_conv2d.sign.data)
                
        #         model._modules[name] = shift_conv2d

        #     conv_count += 1

    return model, conversion_count, linear_count, conv_count

def round_shift_weights(model, clone=False, weight_bits=5, act_integer_bits=16, act_fraction_bits=16):
    if(clone):
        model = copy.deepcopy(model)

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = round_shift_weights(model=module, weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits)

        if type(module) == deepshift.modules.LinearShift or type(module) == deepshift.modules.Conv2dShift:
            module.shift.data = module.shift.round()
            module.sign.data = module.sign.round().sign()

            if (module.bias is not None):
                module.bias.data = utils.round_to_fixed(module.bias, integer_bits=act_integer_bits, fraction_bits=act_fraction_bits)
        elif type(module) == deepshift.modules_q.LinearShiftQ or type(module) == deepshift.modules_q.Conv2dShiftQ:
            module.weight.data = utils.clampabs(module.weight.data, 2**module.shift_range[0], 2**module.shift_range[1]) 
            module.weight.data = utils.round_power_of_2(module.weight)

            if (module.bias is not None):
                module.bias.data = utils.round_to_fixed(module.bias, integer_bits=act_integer_bits, fraction_bits=act_fraction_bits)

    return model

def count_layer_type(model, layer_type):
    count = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(model=module, layer_type=layer_type)
        if type(module) == layer_type:
            count += 1

    return count    


def convert_to_multi(model, weight_bits, rounding, act_integer_bits, act_fraction_bits):
    # linear_count = 0
    # conv_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_to_multi(model=module, weight_bits=weight_bits, rounding=rounding, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits)
        # if conversion_count > 9:
        if type(module) == LinearShift:
            shift_linear = module
            linear = nn.Linear(module.in_features, module.out_features, module.bias is not None) 
            linear.weight.data = utils.get_weight_from_ps(shift_linear.shift, shift_linear.sign, weight_bits, rounding)
            if shift_linear.bias is not None:
                linear.bias.data = utils.get_bias_from_shift(shift_linear.bias, act_integer_bits, act_fraction_bits)
            model._modules[name] = linear

        if type(module) == Conv2dShift:
            shift_conv2d = module
            conv2d = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode) 
            conv2d.weight.data = utils.get_weight_from_ps(shift_conv2d.shift, shift_conv2d.sign, weight_bits, rounding)
            if shift_conv2d.bias is not None:
                conv2d.bias.data = utils.get_bias_from_shift(shift_conv2d.bias, act_integer_bits, act_fraction_bits)
            model._modules[name] = shift_conv2d

    return model