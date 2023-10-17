import torch
import imp
try:
    import unoptimized_cuda
#     # from cuda import unoptimized_cuda
#     from torch.utils.cpp_extension import load
#     unoptimized_cuda = load(
#     'unoptimized_cuda', ['cuda/unoptimized_cuda.cpp', 'cuda/unoptimized.cu'], verbose=True)

except:
    print("Unable to import CUDA unoptimized kernels")

from torch.utils.cpp_extension import load
# unoptimized_cuda = load(
# 'unoptimized_cuda', ['/home/hy34/DeepShift/pytorch/unoptimized/kernels/cuda/unoptimized_cuda.cpp', '/home/hy34/DeepShift/pytorch/unoptimized/kernels/cuda/unoptimized.cu'], verbose=True)

def _import_module_from_library(module_name, path, is_python_module):
    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    file, path, description = imp.find_module(module_name, [path])
    # Close the .so file after load.
    with file:
        if is_python_module:
            return imp.load_module(module_name, file, path, description)
        else:
            torch.ops.load_library(path)

# unoptimized_cuda = _import_module_from_library('unoptimized_cuda', '/tmp/torch_extensions/unoptimized_cuda', True)
# unoptimized_cuda = _import_module_from_library('unoptimized_cuda', '/home/hy34/.cache/torch_extensions/unoptimized_cuda', True)

def linear(input, weight, bias):
    # torch.cuda.set_device(0)
    # out = torch.zeros([input.size(0), input.size(1), weight.size(0)], dtype=torch.float, device=torch.device('cuda:0'))
    out = torch.zeros([input.size(0), input.size(1), weight.size(0)], dtype=torch.float).to("cuda:0")
    if bias is not None:
        # print(input.data.type)
        # print(weight.data.type)
        # print(bias.data.type)
        # print(out.data.type)
        unoptimized_cuda.UNOPTIMIZED_LINEAR(input.float().contiguous(), weight.float().contiguous(), bias.float().contiguous(), out.contiguous())
    else:
        
        temp = torch.zeros([weight.size(0)], dtype=torch.float, device=torch.device('cuda:0'))
        # temp = torch.zeros([weight.size(0)], dtype=torch.float).cuda()
        unoptimized_cuda.UNOPTIMIZED_LINEAR(input, weight, temp, out)

    return out

def conv2d(input, weight, bias, stride, padding):
    if len(stride) == 1:
        strides_h = stride[0]
        strides_w = stride[0]
    else: 
        strides_h = stride[0]
        strides_w = stride[1]
    out_height = int((input.size(2) - weight.size(2)) / strides_h +1)
    out_width = int((input.size(3) - weight.size(3)) / strides_w +1)
    out = torch.zeros([input.size(0), weight.size(0), out_height, out_width], dtype=torch.float, device=torch.device('cuda:0'))
    
    if bias is not None:
        unoptimized_cuda.UNOPTIMIZED_CONV(input.float().contiguous(), weight.float().contiguous(), bias.float().contiguous(), out.float().contiguous(), stride, padding)
    else:
        temp = torch.zeros([weight.size(0)], dtype=torch.float, device=torch.device('cuda:0'))
        unoptimized_cuda.UNOPTIMIZED_CONV(input, weight, temp, out, stride, padding )

    return out