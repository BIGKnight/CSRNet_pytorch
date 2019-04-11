import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module
import dilated_conv2d_gpu as dilated_conv2d


class DilatedConvolutionFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 7:
            print("wrong input parameters number, check the input")
            return
        input = args[0]
        weights = args[1]
        bias = args[2]
        ctx.stride_h = args[3]
        ctx.stride_w = args[4]
        ctx.dilation_h = args[5]
        ctx.dilation_w = args[6]
        output = dilated_conv2d.forward(input, weights, bias, ctx.stride_h, ctx.stride_w, ctx.dilation_h, ctx.dilation_w)
        ctx.save_for_backward(input, weights, bias)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) != 1:
            print("Wrong output number, check your output")
            return
        input, weights, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = dilated_conv2d.backward(input, weights, bias, grad_outputs[0], ctx.stride_h, ctx.stride_w, ctx.dilation_h, ctx.dilation_w)
        return grad_input, grad_weight, grad_bias, None, None, None, None


class DilatedConv2dLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_h, stride_w, dilation_h, dilation_w):
        super(DilatedConv2dLayer, self).__init__()
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight, gain=1)


    def forward(self, inputs):
        return DilatedConvolutionFunction.apply(inputs, self.weight, self.bias, self.stride_h, self.stride_w, self.dilation_h, self.dilation_w)

class BasicDilatedConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(BasicDilatedConv2D, self).__init__()
        self.pad = padding
        self.dilated_conv2d = DilatedConv2dLayer(in_channels, out_channels, kernel_size, 1, 1, dilation, dilation)
        
    def forward(self, x):
        x = nn.functional.pad(x, [self.pad, self.pad, self.pad, self.pad])
        return self.dilated_conv2d(x)