import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append("/home/hy34/DeepShift/pytorch")

from unoptimized.modules.linear import UnoptimizedLinear
from unoptimized.modules.conv import UnoptimizedConv2d

def convert_to_unoptimized(model, linear_count=0, conv_count=0):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], linear_count, conv_count = convert_to_unoptimized(model=module, linear_count=linear_count, conv_count=conv_count)
        if type(module) == nn.Linear:
            linear = module     
            if name != 'head' and linear_count > 8:  
                unoptimized_linear = UnoptimizedLinear(module.in_features, module.out_features, module.bias is not None) 
                unoptimized_linear.weight = linear.weight
                unoptimized_linear.bias = linear.bias
                model._modules[name] = unoptimized_linear
            linear_count += 1

        if type(module) == nn.Conv2d:
            conv2d = module
            if (name != "proj") and conv_count > 1:
                unoptimized_conv = UnoptimizedConv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                                    module.padding, module.dilation, module.groups,
                                                    module.bias is not None, module.padding_mode) 
                unoptimized_conv.bias = conv2d.bias
                unoptimized_conv.weight = conv2d.weight

                model._modules[name] = unoptimized_conv
            conv_count += 1

    return model, linear_count, conv_count
    
    
if __name__ == '__main__':
    # this test will be run if you type in the command:
    # > python convert_to_unoptimized    
    import torchvision.models as models
    model = models.__dict__['resnet18'](pretrained=True)
    model = model.to("cuda:0")
    input = torch.rand((32, 3, 224, 224)).to("cuda:0")
    output1 = model(input)


    model = convert_to_unoptimized(model).to("cuda:0")
    output2 = model(input)

    max_error = torch.max(torch.abs(output1 - output2))
    print(max_error.detach().cpu().numpy())

