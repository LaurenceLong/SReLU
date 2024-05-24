import torch
from torch import nn

import importlib.util

if importlib.util.find_spec('srelu_cuda'):
    import srelu_cuda
else:
    print("SReLU_cuda is not available, you can use it after compile")


class SReLU_cuda(nn.Module):
    """
    CUDA version, native compile required
    usage: Refer to ./native/INSTALL.md
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, t=2.21, inplace=False):
        super(SReLU_cuda, self).__init__()
        self.t = t
        self.a = (torch.pi / 2) / t
        self.inplace = inplace

    def forward(self, x):
        return SReLUFunction.apply(x, self.t, self.a, self.inplace)


class SReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, t, a, inplace):
        ctx.t = t
        ctx.a = a
        ctx.save_for_backward(x)

        return srelu_cuda.forward(x, t, a, inplace)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        t = ctx.t
        a = ctx.a
        grad_input = srelu_cuda.backward(grad_output, x, t, a)
        return grad_input, None, None, None


SReLU_cuda()


class SReLU(nn.Module):
    """
    # Non CUDA version
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, t=2.21, inplace=False):
        super(SReLU, self).__init__()
        self.t = t = torch.tensor(t, dtype=torch.float32)
        self.a = (torch.pi / 2) / t

        self.inplace = inplace
        self.scripted_forward = torch.jit.script(self._forward)

    def forward(self, x):
        # 使用已编译的函数版本
        return self.scripted_forward(x, self.t, self.a, self.inplace)

    @staticmethod
    def _forward(x, t, a, inplace: bool = False):
        # 使用掩码 (mask) 的方法实现 TReLU
        mask_l = x <= -t
        mask_r = x <= t
        mask_mid = ~mask_l & mask_r

        if inplace:
            out = x
        else:
            out = torch.empty_like(x)
            out[~mask_r] = x[~mask_r]  # x > 0, y = x

        out[mask_l] = 0  # 置0
        out[mask_mid] = x[mask_mid] * (torch.sin(a * x[mask_mid]) + 1) / 2  # 中间区域靠左

        return out
