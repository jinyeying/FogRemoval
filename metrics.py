from math import exp, log10
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from losses import PerceptualLoss

class ImageReconstructionError(nn.Module):
    def __init__(self, metrics=['psnr', 'ssim']):
        super(ImageReconstructionError, self).__init__()
        for metric in metrics:
            if metric.lower() == 'mse':
                self.mse = nn.MSELoss()
            elif metric.lower() == 'mae':
                self.mae = nn.L1Loss()
            elif metric.lower() == 'psnr':
                self.psnr = nn.MSELoss()
            elif metric.lower() == 'ssim':
                self.ssim = SSIM()
            elif metric.lower() == 'perc':
                self.perc = PerceptualLoss()
            else:
                raise NotImplementedError('metric [%s] is not found' % metric)

    def forward(self, tensor_eval, tensor_ref, metric):
        """
        both input and reference tensors should be a minibatch, 4D tensor
        Also, assuming values ranges in [0, 1]
        """
        assert len(tensor_eval.shape) == 4
        assert len(tensor_ref.shape) == 4
        assert (0. <= tensor_eval).all() and (tensor_eval <= 1.).all()
        assert (0. <= tensor_ref).all() and (tensor_ref <= 1.).all()
        assert tensor_eval.shape[0] == tensor_ref.shape[0]

        metric_layer = getattr(self, metric.lower())

        batch_size = tensor_eval.shape[0]
        out = 0.
        for bid in range(batch_size):
            val = metric_layer(tensor_eval[bid:bid+1], tensor_ref[bid:bid+1]).item()
            if metric.lower() == 'psnr':
                out += 10 * log10(1. / val)
            else:
                out += val
        return out / float(batch_size)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, auto_downsample=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.auto_downsample = auto_downsample
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, height, weight) = img1.size()

        f = max(1., round(min(height, weight)/256.))
        if self.auto_downsample and f > 1:
            img1 = F.avg_pool2d(img1, f)
            img2 = F.avg_pool2d(img2, f)

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
