import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def CircularGaussKernel(kernlen=None, circ_zeros=False, sigma=None, norm=True):
    assert ((kernlen is not None) or sigma is not None)
    if kernlen is None:
        kernlen = int(2.0 * 3.0 * sigma + 1.0)
        if kernlen % 2 == 0:
            kernlen = kernlen + 1
        halfSize = kernlen / 2
    halfSize = kernlen / 2
    r2 = float(halfSize*halfSize)
    if sigma is None:
        sigma2 = 0.9 * r2
        sigma = np.sqrt(sigma2)
    else:
        sigma2 = 2.0 * sigma * sigma
    x = np.linspace(-halfSize,halfSize,kernlen)
    xv, yv = np.meshgrid(x, x, sparse=False, indexing='xy')
    distsq = xv**2 + yv**2
    kernel = np.exp(-(distsq / sigma2))
    if circ_zeros:
        kernel *= (distsq <= r2).astype(np.float32)
    if norm:
        kernel /= np.sum(kernel)
    return kernel


class GaussianBlur(nn.Module):
    def __init__(self, sigma=1.6):
        super(GaussianBlur, self).__init__()
        weight = self.calculate_weights(sigma)
        self.register_buffer('buf', weight)
        return

    def calculate_weights(self, sigma):
        kernel = CircularGaussKernel(sigma=sigma, circ_zeros=False)
        h,w = kernel.shape
        halfSize = float(h) / 2.
        self.pad = int(np.floor(halfSize))
        return torch.from_numpy(kernel.astype(np.float32)).view(1, 1, h, w)

    def forward(self, x):
        w = Variable(self.buf)
        return F.conv2d(F.pad(x, (self.pad, self.pad, self.pad, self.pad), 'replicate'),
                        w.repeat(x.shape[1], 1, 1, 1), padding=0, groups=x.shape[1])

class SimpleGray(nn.Module):
    def __init__(self):
        super(SimpleGray, self).__init__()
        gray_vector = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        self.register_buffer('buf', gray_vector)
        return

    def forward(self, x):
        w = Variable(self.buf)
        return F.conv2d(x, w, padding=0)


class RGB2Saturation(nn.Module):
    def __init__(self):
        super(RGB2Saturation, self).__init__()

    def forward(self, x):
        # match range
        x = (x + 1.) / 2.

        x_min, _ = torch.min(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_sat = 1. - x_min / x_max.clamp(min=1e-8)
        x_sat = x_sat * 2. - 1.
        return x_sat

