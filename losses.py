import torch
from torch import nn
from torchvision.models.vgg import vgg16, vgg19
from modules import SimpleGray

class PixelwiseLoss(nn.Module):
    """
    It is just a simple MSE loss
    assuming input in range [-1, 1]
    """
    def __init__(self, is_gray=False):
        super(PixelwiseLoss, self).__init__()
        if is_gray:
            self.gray_layer = SimpleGray()
        self.criterion = nn.MSELoss()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, out_images, target_images, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            if hasattr(self, 'gray_layer'):
                out_images = self.gray_layer((out_images + 1.) / 2.)
                target_images = self.gray_layer((target_images + 1.) / 2.)
            return weight * self.criterion(out_images, target_images)

class PixelwiseGrayLoss(nn.Module):
    """
    It is just a simple MSE loss
    assuming input in range [-1, 1]
    """
    def __init__(self, is_gray=False):
        super(PixelwiseGrayLoss, self).__init__()
        if is_gray:
            self.gray_layer = SimpleGray()
        self.criterion = nn.MSELoss()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, out_images, target_images, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            out_images = torch.mean(out_images,dim=1,keepdim= True)
            target_images = torch.mean(target_images,dim=1,keepdim= True)
            return weight * self.criterion(out_images, target_images)

class PerceptualLoss(nn.Module):
    def __init__(self, model='vgg19_5_4', include_max_pool=False, norm=False, is_gray=False):
        super(PerceptualLoss, self).__init__()
        if model.startswith('vgg16'):
            # vgg16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
            #          1   3    5    6    8   10   11   13   15   17   18   20   22   24   25   27   29   31
            vgg = vgg16(pretrained=True)
            if model.endswith('2_2'):
                layer_idx, output_nc = 10, 128
            elif model.endswith('5_3'):
                layer_idx, output_nc = 31, 512
            else:
                raise NotImplementedError('Only support [2_2 and 5_3]')
        elif model.startswith('vgg19'):
            # vgg19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
            #          1   3    5    6    8   10   11   13   15   17   19   20   22   24   26   28   29   31   33   35   37
            vgg = vgg19(pretrained=True)
            if model.endswith('2_2'):
                layer_idx, output_nc = 10, 128
            elif model.endswith('5_4'):
                layer_idx, output_nc = 37, 512
            else:
                raise NotImplementedError('Only support [2_2 and 5_4]')

        if not include_max_pool:
            layer_idx -= 1
        loss_network = nn.Sequential(*list(vgg.features)[:layer_idx]).eval()

        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network

        # input normalization
        t_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        t_mean -= 1.                                           # if input in range [-1, 1]
        t_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t_std *= 2.                                            # if input in range [-1, 1]
        self.register_buffer('mean', t_mean)
        self.register_buffer('std', t_std)

        # criterion
        self.mse_loss = nn.MSELoss()

        # domain-invariant perceptual loss from MUNIT paper
        # seems not working... T^T
        if norm:
            self.norm_layer = nn.InstanceNorm2d(output_nc)

        if is_gray:
            self.gray_layer = SimpleGray()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, out_images, target_images, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            # make it gray
            if hasattr(self, 'gray_layer'):
                out_images = self.gray_layer((out_images + 1.) / 2.).expand_as(out_images)
                target_images = self.gray_layer((target_images + 1.) / 2.).expand_as(target_images)

            # normalization
            out_images = (out_images - self.mean) / self.std
            target_images = (target_images - self.mean) / self.std

            out_features = self.loss_network(out_images)
            target_feature = self.loss_network(target_images)
            if hasattr(self, 'norm_layer'):
                return weight * self.mse_loss(self.norm_layer(out_features), self.norm_layer(target_feature))
            else:
                return weight * self.mse_loss(out_features, target_feature)

class PerceptualMultiplierLoss(nn.Module):
    def __init__(self, model='vgg19_5_4', include_max_pool=False, norm=False, is_gray=False):
        super(PerceptualMultiplierLoss, self).__init__()

        # vgg19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        #          1   3    5    6    8   10   11   13   15   17   19   20   22   24   26   28   29   31   33   35   37
        vgg = vgg19(pretrained=True)
      
        layer_idx, output_nc = 19, 256

        if not include_max_pool:
            layer_idx -= 1
        loss_network = nn.Sequential(*list(vgg.features)[:layer_idx]).eval()

        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network

        # input normalization
        t_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        t_mean -= 1.                                           # if input in range [-1, 1]
        t_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t_std *= 2.                                            # if input in range [-1, 1]
        self.register_buffer('mean', t_mean)
        self.register_buffer('std', t_std)

        # criterion
        self.mse_loss = nn.MSELoss()

        if norm:
            self.norm_layer = nn.InstanceNorm2d(output_nc)

        if is_gray:
            self.gray_layer = SimpleGray()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, out_images, target_images, multiplier, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            # make it gray

            out_images = torch.mean(out_images,dim=1,keepdim= True).expand_as(out_images)
            target_images  = torch.mean(target_images,dim=1, keepdim= True).expand_as(target_images)                

            # normalization
            out_images = (out_images - self.mean) / self.std
            target_images = (target_images - self.mean) / self.std


            out_features = self.loss_network(out_images)
            target_feature = self.loss_network(target_images)

            if hasattr(self, 'norm_layer'):
                return weight * self.mse_loss(self.norm_layer(out_features), self.norm_layer(target_feature * multiplier.detach() ) )
            else:
                return weight * self.mse_loss(out_features, target_feature )

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

        if torch.cuda.is_available():
            self.cuda()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class CrossEntropyGANLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyGANLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.cuda()

    def get_target_tensor(self, input, label):
        label_tensor = torch.tensor(label, dtype=torch.long)
        if torch.cuda.is_available():
            label_tensor = label_tensor.cuda()
        return label_tensor.expand(input.size()[:1] + input.size()[2:])

    def __call__(self, input, label):
        target = self.get_target_tensor(input, label)
        return self.loss(input, target)

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class EnTVLoss(nn.Module):
    def __init__(self):
        super(EnTVLoss, self).__init__()
        self.eps = 1/255.
        self.sigma = 6.0 * self.eps

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x, y=None, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            if y is not None:
                ref_gx, ref_gy = self.calc_gradxy(y)
                wx = torch.exp(-ref_gx.pow_(2) / (2 * self.sigma ** 2))
                wy = torch.exp(-ref_gy.pow_(2) / (2 * self.sigma ** 2))
            else:
                wx, wy = 1., 1.

            tvx, tvy = self.calc_gradxy(x)
            tvx = (torch.pow(tvx, 2) * wx).mean()
            tvy = (torch.pow(tvy, 2) * wy).mean()
            return weight * 2 * (tvx + tvy)

    @staticmethod
    def calc_gradxy(t):
        gx = t[:, :, 1:, :] - t[:, :, :-1, :]
        gy = t[:, :, :, 1:] - t[:, :, :, :-1]
        return gx, gy


class EqTVLoss(nn.Module):
    def __init__(self):
        super(EqTVLoss, self).__init__()
        self.criterion = nn.MSELoss()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, pred, trans, input_img, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            pred = (pred + 1.) / 2.
            trans = (trans + 1.) / 2.
            input_img = (input_img + 1.) / 2.

            gxi, gyi = self.calc_gradxy(input_img)
            gxj, gyj = self.calc_gradxy(pred*trans)

            return weight * 2 * (self.criterion(gxj, gxi) + self.criterion(gyj, gyi))

    @staticmethod
    def calc_gradxy(t):
        gx = t[:, :, 1:, :] - t[:, :, :-1, :]
        gy = t[:, :, :, 1:] - t[:, :, :, :-1]
        return gx, gy


class GrayLoss(nn.Module):
    def __init__(self, weighted_average=False, abs_cos=False):
        super(GrayLoss, self).__init__()
        self.weighted_average = weighted_average
        self.register_buffer('gray', torch.ones((3,)).view(1, 3, 1, 1))
        self.zero = torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        self.cosine_sim = nn.CosineSimilarity()
        self.abs_cos = abs_cos

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x, weight=1.0):
        if (weight == 0.) or (x.shape[1] < 2):
            return self.zero
        else:
            # pixelwise consine similarity
            cosine_sim = self.cosine_sim(x, self.gray)

            # pixelwise cosine embedding loss
            if self.abs_cos:
                cos_embed_loss = 1 - cosine_sim.abs()
            else:
                cos_embed_loss = (1 - cosine_sim)/2

            if self.weighted_average:
                x_norm = torch.norm(x, 2, dim=1)
                loss = (x_norm * cos_embed_loss).sum() / x_norm.sum()
            else:
                loss = cos_embed_loss.mean()

            return weight * loss

class CenterLoss(nn.Module):

    def __init__(self):
        super(CenterLoss, self).__init__()
        self.criterion = nn.MSELoss()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x, weight=1.0):
        if weight == 0. or (x.size(2) == 1 and x.size(3) == 1):
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            bs, cs = x.size(0), x.size(1)
            assert cs == 3
            x = x.view((bs, cs, -1))
            assert x.size(2) > 1
            xmean = x.mean(dim=-1, keepdim=True)
            return weight * (x - xmean).pow(2).sum(dim=1).sqrt().mean()

class HazelineLoss(nn.Module):
    def __init__(self, use_chromaticity=False, weighted_average=False, abs_cos=False, norm_input=False, mask=False):
        super(HazelineLoss, self).__init__()
        self.use_chromaticity = use_chromaticity
        self.weighted_average = weighted_average
        self.abs_cos = abs_cos
        self.norm_input= norm_input
        self.mask = mask

        self.cosine_sim = nn.CosineSimilarity()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, hazy_input, pred, airlight, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            hazy_input = (hazy_input + 1.) / 2.
            pred = (pred + 1.) / 2.

            mask = 1.
            if self.norm_input:
                airlight = torch.ones_like(airlight).cuda() if torch.cuda.is_available() else torch.ones_like(airlight)

            if self.use_chromaticity:
                hazy_input = hazy_input / hazy_input.sum(dim=1, keepdim=True).clamp(min=1e-8)
                pred = pred / pred.sum(dim=1, keepdim=True).clamp(min=1e-8)
                airlight = airlight / 3.
                if self.mask:
                    mask *= ((hazy_input - airlight).norm(p=2, dim=1, keepdim=True) > 0.01).float()
            else:
                # by intensity
                if self.mask:
                    mask *= ((hazy_input - airlight).norm(p=2, dim=1, keepdim=True) > 0.25).float()

            # pixelwise cosine similarity, [-pi, pi]
            cosine_sim = self.cosine_sim(hazy_input - airlight, pred - airlight)
            if self.mask:
                mask *= (cosine_sim.unsqueeze_(1) > 0.0).float()

            # pixelwise cosine embedding loss
            if self.abs_cos:
                cos_embed_loss = 1 - cosine_sim.abs()
            else:
                cos_embed_loss = (1 - cosine_sim)/2

            loss = (cos_embed_loss * mask).mean()

            return weight * loss

class DistantLoss(nn.Module):
    def __init__(self):
        super(DistantLoss, self).__init__()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, pred, airlight, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            pred = (pred + 1.) / 2.
            janorm = torch.norm(pred - airlight, 2, dim=1)
            loss = torch.exp(-janorm).mean()

            return weight * loss

class DistantPreserveLoss(nn.Module):
    def __init__(self):
        super(DistantPreserveLoss, self).__init__()
        self.criterion = nn.MSELoss()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, pred, pre_pred, airlight, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            pred = (pred + 1.) / 2.
            pre_pred = (pre_pred + 1.) / 2.

            norm_curr = torch.norm(pred - airlight, 2, dim=1)
            norm_prev = torch.norm(pre_pred - airlight, 2, dim=1)
            loss = self.criterion(norm_curr, norm_prev)

            return weight * loss

class DistancePreserveLoss(nn.Module):
    def __init__(self, use_chromaticity=False):
        super(DistancePreserveLoss, self).__init__()
        self.criterion = nn.MSELoss()

        self.use_chromaticity = use_chromaticity

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, pred, pre_pred, airlight, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            pred = (pred + 1.) / 2.
            pre_pred = (pre_pred + 1.) / 2.

            if self.use_chromaticity:
                pred = pred / pred.sum(dim=1, keepdim=True).clamp(min=1e-8)
                pre_pred = pre_pred / pre_pred.sum(dim=1, keepdim=True).clamp(min=1e-8)
                airlight = airlight / airlight.sum(dim=1, keepdim=True).clamp(min=1e-8)

            norm_curr = torch.norm(pred - airlight, 2, dim=1)
            norm_prev = torch.norm(pre_pred - airlight, 2, dim=1)
            loss = self.criterion(norm_curr, norm_prev)

            return weight * loss

class SaturationPreserveLoss(nn.Module):
    def __init__(self):
        super(SaturationPreserveLoss, self).__init__()
        self.criterion = nn.MSELoss()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, pred, pre_pred, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            pred = (pred + 1.) / 2.
            pre_pred = (pre_pred + 1.) / 2.

            pred_sat = 1. - pred.min(dim=1)[0] / pred.max(dim=1)[0].clamp(min=1e-8)
            pre_pred_sat = 1. - pre_pred.min(dim=1)[0] / pre_pred.max(dim=1)[0].clamp(min=1e-8)

            loss = self.criterion(pred_sat, pre_pred_sat)

            return weight * loss