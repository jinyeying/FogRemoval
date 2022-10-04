import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import functools
from random import getrandbits
from modules import SimpleGray, GaussianBlur, RGB2Saturation
from losses import GANLoss
import torch.nn.functional as F

###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'batch_no_track':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'instance_affine':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'instance_track':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    elif norm_type == 'group':
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, config,
             init_type='normal', use_dropout=False, final_activation='tanh', gpu_ids=[]):
    which_model_netG = config['model']
    ngf = config['dim']
    norm_layer = get_norm_layer(norm_type=config['norm'])

    if which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer,
                             use_dropout=use_dropout, final_activation=final_activation)
    elif which_model_netG == 'unet_64':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer,
                             use_dropout=use_dropout, final_activation=final_activation)
    elif which_model_netG == 'n_layers':
        netG = NLayerEstimator(input_nc, output_nc, ngf, n_layers=config['n_layers'],
                               norm_layer=norm_layer, final_activation=final_activation)
    elif which_model_netG == 'unet_256_new':
        netG = UnetGenerator_new(input_nc, output_nc, 8, ngf, norm_layer=norm_layer,
                             use_dropout=use_dropout, final_activation=final_activation)   
    elif which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator_new(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)    
    elif which_model_netG == 'resnet_9blocks_K':
        netG = ResnetGenerator_K(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)       
    elif which_model_netG == 'pix2pixHD':   
        n_downsample_global=3
        n_blocks_global=9
        n_local_enhancers=1
        n_blocks_local=3     
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)   
    elif which_model_netG == 'pix2pix6HD':   
        n_downsample_global=3
        n_blocks_global=6
        n_local_enhancers=1
        n_blocks_local=3     
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)                                  
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    netG = init_net(netG, init_type, gpu_ids)
    if torch.cuda.is_available():
        netG = nn.DataParallel(netG).cuda()
    return netG

def define_GH(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_9blocks_new':  
        netG = ResnetGenerator_new(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)                                   
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    netG = init_net(netG, init_type, gpu_ids)
    if torch.cuda.is_available():
        netG = nn.DataParallel(netG).cuda()

    return netG   


def define_D(input_nc, config, init_type='normal', gpu_ids=[]):
    ndf = config['dim']
    which_model_netD = config['model']
    n_layers_D = config['n_layers']
    use_sigmoid = (not config['use_lsgan'])
    final_activation = 'sigmoid' if use_sigmoid else None

    netD = None
    norm_layer = get_norm_layer(norm_type=config['norm'])

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'unet_256':
        netD = UnetGenerator(input_nc, 1, 8, ndf, norm_layer=norm_layer, use_dropout=False, final_activation=final_activation)
    elif which_model_netD == 'SRGAN':
        # stride [2, 2, 2, 2] 22x22 before fc
        netD = NLayerEstimator(input_nc, 1, ndf, n_layers=3, norm_layer=norm_layer, final_activation=final_activation)
    elif which_model_netD == 'DPED':
        # stride [4, 2, 2] 7x7 before fc
        netD = NLayerEstimator(input_nc, 1, ndf, n_layers=4, norm_layer=norm_layer, final_activation=final_activation)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)

    return init_net(netD, init_type, gpu_ids)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetGenerator_new(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_new, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_en = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model_en += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model_en += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model_de = []    

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_de += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model_de += [nn.ReflectionPad2d(3)]
        model_de += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_de += [nn.Tanh()]

        self.model_en = nn.Sequential(*model_en)
        self.model_de = nn.Sequential(*model_de)

    def forward(self, input):
        feat = self.model_en(input)   
        out = self.model_de(feat)   
        return out      

class ResnetGenerator_K(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_K, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_en = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model_en += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model_en += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model_K = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model_de = []    

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_de += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model_de += [nn.ReflectionPad2d(3)]
        model_de += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_de += [nn.Tanh()]

        self.model_en = nn.Sequential(*model_en)
        self.model_de = nn.Sequential(*model_de)
        self.res_K = nn.Sequential(*model_K)

    def forward(self, input):
        feat = self.model_en(input) 
        K = self.res_K(feat)
        feat_new =  feat * K
        out = self.model_de(feat)   
        return out               

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, final_activation='tanh'):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, final_activation=final_activation)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_new(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, final_activation='tanh'):
        super(UnetGenerator_new, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)


        self.unet_block1 = UnetSkipBlocks(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.unet_block2 = UnetSkipBlocks(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.unet_block3 = UnetSkipBlocks(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer, final_activation=final_activation)

        self.unet_block0 = unet_block

    def forward(self, input):
        feat_down3 = self.unet_block3.down(input)
        feat_down2 = self.unet_block2.down(feat_down3)
        feat_down1 = self.unet_block1.down(feat_down2)
        feat0 = self.unet_block0(feat_down1)
        feat_up1 = self.unet_block1.up(feat0)
        feat_up2 = self.unet_block2.up(torch.cat([feat_down2, feat_up1], 1))
        out =      self.unet_block3.up(torch.cat([feat_down3, feat_up2], 1))
        return out

    def forward_front(self, input):
        feat_down3 = self.unet_block3.down(input)
        feat_down2 = self.unet_block2.down(feat_down3)
        feat_down1 = self.unet_block1.down(feat_down2)
        feat0 = self.unet_block0(feat_down1)
        return feat0, feat_down2, feat_down3

    def forward_back(self, feat0, feat_down2, feat_down3):
        feat_up1 = self.unet_block1.up(feat0)
        feat_up2 = self.unet_block2.up(torch.cat([feat_down2, feat_up1], 1))
        out =      self.unet_block3.up(torch.cat([feat_down3, feat_up2], 1))
        return out        

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, final_activation='tanh',
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            if final_activation == 'tanh':
                up += [nn.Tanh()]
            elif final_activation == 'relu':
                up += [nn.ReLU()]
            elif final_activation == 'sigmoid':
                up += [nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetSkipBlocks(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, final_activation='tanh',
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipBlocks, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            if final_activation == 'tanh':
                up += [nn.Tanh()]
            elif final_activation == 'relu':
                up += [nn.ReLU()]
            elif final_activation == 'sigmoid':
                up += [nn.Sigmoid()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
        self.up = nn.Sequential(*up)
        self.down = nn.Sequential(*down)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class NLayerEstimator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, final_activation='sigmoid'):
        super(NLayerEstimator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.AdaptiveAvgPool2d(1),
                     nn.Conv2d(ndf * nf_mult, 1024, kernel_size=1),
                     nn.LeakyReLU(0.2, True),
                     nn.Conv2d(1024, output_nc, kernel_size=1)]

        if final_activation == 'sigmoid':
            sequence += [nn.Sigmoid()]
        elif final_activation == 'relu':
            sequence += [nn.ReLU()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class GenerativeModel(nn.Module):
    def __init__(self, config):
        super(GenerativeModel, self).__init__()

        if config['image_model'] == 'direct':
            self.netG = define_G(3, 3, config['netG'])
        elif config['image_model'] == 'k':
            self.netG = define_G(3, 3, config['netG'], final_activation='relu')
        elif config['image_model'].startswith('t'):
            output_nc = 3 if config['image_model'].startswith('t3') else 1
            self.netG = define_G(3, output_nc, config['netG'])
        elif config['image_model'].startswith('J'):
            self.netG = define_G(3, 3, config['netG'])
        elif config['image_model'].startswith('G'):
            self.netG = define_G(1, 1, config['netG'])    
        else:
            raise NotImplementedError('image model [%s] is not found' % config['image_model'])

        if not config['image_model'].endswith('AnalyticA'):
            if 'Av-' in config['image_model']:
                self.netA = define_G(3, 3, config['netA'], final_activation='sigmoid')
            elif 'Am-' in config['image_model']:
                self.netA = define_G(3, 3, config['netA'], final_activation='sigmoid')

        self.config = config

    def forward(self, x):
        _I = (x + 1.) / 2.                                                      # to [0,1]
        eps = 1e-5

        if self.config['image_model'] == 'direct':
            return self.netG(x), [None, None]
        elif self.config['image_model'] == 'k':
            _K = self.netG(x)                                                   # [0,inf] relu
            _J = _K*_I-_K+1.
            return _J*2.-1., [(_K+1.).log_()-1., None]                          # [-1,1]

        elif self.config['image_model'].startswith('t'):
            if self.config['image_model'].endswith('-parallel'):
                if self.config['alter_grad_up']:
                    self.rand_alter_grad_up(self.netA, self.netG)
                _A, _t = self.netA(x), self.netG(x)
            elif self.config['image_model'].endswith('-A-first'):
                _A = self.netA(x)
                _t = self.netG(_I-_A)
            elif self.config['image_model'].endswith('-t-first'):
                _t = self.netG(x)
                _A = self.netA(_t*x)
            elif self.config['image_model'].endswith('AnalyticA'):
                _A = self.estimate_airlight(_I.detach(), self.config['A_method'])
                if self.config['norm_input'] == 'chr':
                    _A_chr = _A / _A.sum(dim=1, keepdim=True)  
                    x = _I / _A_chr / 3.  
                elif self.config['norm_input'] == 'rgb':
                    x = _I / _A

                _t = self.netG(x * 2. - 1.)
            else:
                raise NotImplementedError('image model [%s] is not found' % self.config['image_model'])

            _t = torch.clamp((_t+1.)/2., eps, 1.)                               # [0,1]
            _J = ((_I-_A)/_t + _A)
            if self.config['clamp_latent']:
                _J = torch.clamp(_J, 0., 1.)
            return _J*2.-1., [_t*2.-1., _A]                                     # [-1,1]

        elif self.config['image_model'].startswith('J') or self.config['image_model'].startswith('G'):
            if self.config['image_model'].endswith('-parallel'):
                if self.config['alter_grad_up']:
                    self.rand_alter_grad_up(self.netA, self.netG)
                _A, _J = self.netA(x), self.netG(x)
            elif self.config['image_model'].endswith('-A-first'):
                _A = self.netA(x)
                _J = self.netG(_I-_A)
            elif self.config['image_model'].endswith('AnalyticA'):
                _A = self.estimate_airlight(_I.detach(), self.config['A_method'])
                if self.config['norm_input'] == 'chr':
                    _A_chr = _A / _A.sum(dim=1, keepdim=True)  
                    x = _I / _A_chr / 3.  
                elif self.config['norm_input'] == 'rgb':
                    x = _I / _A

                _J = self.netG(x * 2. - 1.)
            else:
                raise NotImplementedError('image model [%s] is not found' % self.config['image_model'])

            _J = (_J + 1.) / 2.
            # cliping _J for safe division
            _J = torch.where((_J-_A).abs() <= eps, ((_J >= _A).float()*2.-1.)*eps + _A, _J)
            _t = (_I - _A) / (_J - _A)
            if self.config['clamp_latent']:
                _t = torch.clamp(_t, eps, 1.)
            return _J*2.-1., [_t*2.-1., _A]                                     # [-1,1]
        else:
            raise NotImplementedError('image model [%s] is not found' % self.config['image_model'])

    def forward_At(self, x, netG_x):
        _I = (x + 1.) / 2.                                                      # to [0,1]
        eps = 1e-5

        # output J, [t, A] if available else None
        if self.config['image_model'] == 'direct':
            return netG_x, [None, None]
        elif self.config['image_model'] == 'k':
            _K = netG_x                                                   # [0,inf] relu
            _J = _K*_I-_K+1.
            return _J*2.-1., [(_K+1.).log_()-1., None]                          # [-1,1]

        elif self.config['image_model'].startswith('t'):
            if self.config['image_model'].endswith('-parallel'):
                if self.config['alter_grad_up']:
                    self.rand_alter_grad_up(self.netA, self.netG)
                _A, _t = self.netA(x), netG_x
            elif self.config['image_model'].endswith('-A-first'):
                _A = self.netA(x)
                _t = self.netG(_I-_A)
            elif self.config['image_model'].endswith('-t-first'):
                _t = netG_x
                _A = self.netA(_t*x)
            elif self.config['image_model'].endswith('AnalyticA'):
                _A = self.estimate_airlight(_I.detach(), self.config['A_method'])
                if self.config['norm_input'] == 'chr':
                    _A_chr = _A / _A.sum(dim=1, keepdim=True)  
                    x = _I / _A_chr / 3.  # roughly [0, 1]
                elif self.config['norm_input'] == 'rgb':
                    x = _I / _A

                _t = self.netG(x * 2. - 1.)
            else:
                raise NotImplementedError('image model [%s] is not found' % self.config['image_model'])

            _t = torch.clamp((_t+1.)/2., eps, 1.)                               # [0,1]
            _J = ((_I-_A)/_t + _A)
            if self.config['clamp_latent']:
                _J = torch.clamp(_J, 0., 1.)
            return _J*2.-1., [_t*2.-1., _A]                                     # [-1,1]

        elif self.config['image_model'].startswith('J'):
            if self.config['image_model'].endswith('-parallel'):
                if self.config['alter_grad_up']:
                    self.rand_alter_grad_up(self.netA, self.netG)
                _A, _J = self.netA(x), netG_x
            elif self.config['image_model'].endswith('-A-first'):
                _A = self.netA(x)
                _J = self.netG(_I-_A)
            elif self.config['image_model'].endswith('AnalyticA'):
                _A = self.estimate_airlight(_I.detach(), self.config['A_method'])
                if self.config['norm_input'] == 'chr':
                    _A_chr = _A / _A.sum(dim=1, keepdim=True)  
                    x = _I / _A_chr / 3.  # roughly [0, 1]
                elif self.config['norm_input'] == 'rgb':
                    x = _I / _A

                _J = self.netG(x * 2. - 1.)
            else:
                raise NotImplementedError('image model [%s] is not found' % self.config['image_model'])

            _J = (_J + 1.) / 2.
            # cliping _J for safe division
            _J = torch.where((_J-_A).abs() <= eps, ((_J >= _A).float()*2.-1.)*eps + _A, _J)
            _t = (_I - _A) / (_J - _A)
            if self.config['clamp_latent']:
                _t = torch.clamp(_t, eps, 1.)
            return _J*2.-1., [_t*2.-1., _A]                                     # [-1,1]
        else:
            raise NotImplementedError('image model [%s] is not found' % self.config['image_model'])            

    @staticmethod
    def rand_alter_grad_up(net_a, net_b):
        # random alternative gradient update
        up_or_not = bool(getrandbits(1))
        for p in net_a.parameters():
            p.requires_grad = up_or_not
        for p in net_b.parameters():
            p.requires_grad = not up_or_not

    @staticmethod
    def estimate_airlight(_I, method):
        # atmospheric light color vector
        bs, ch, h, w = _I.shape[0], _I.shape[1], _I.shape[2], _I.shape[3]
        percentile = int(round(h*w*0.001))

        if method == 'he':
            wsz = 7     

            # proper image size for accurate local operation
            f = max(1., round(min(h, w) / 256.))
            if f > 1:
                # simple low-pass filter
                _I = nn.functional.avg_pool2d(_I, f)

            darkchannel, _ = torch.min(_I, dim=1, keepdim=True)
            darkchannel = 1. - nn.functional.max_pool2d(1.-darkchannel, wsz, stride=1, padding=wsz//2)

            darkchannel = darkchannel.view([bs, 1, -1])
            srt_val, _ = torch.sort(darkchannel, descending=True)
            hazy_pixels = darkchannel >= srt_val[:, :, percentile: percentile+1]

            _I_brightness = (_I ** 2.).sum(dim=1, keepdim=True).sqrt().view([bs, 1, -1])
            _I_brightness = _I_brightness * hazy_pixels.float()
            brightest_pixel, _ = torch.max(_I_brightness, dim=2, keepdim=True)
            brightest_pixel_location = (_I_brightness == brightest_pixel).float()
            _A = _I.view([bs, 3, -1]) * brightest_pixel_location
            _A = _A.sum(dim=-1) / brightest_pixel_location.sum(dim=-1)
            return _A.view([bs, 3, 1, 1])

        elif method == 'sp_cc':
            input_pixels = _I.view(bs, _I.shape[1], -1)  
            norm_vec = torch.norm(input_pixels.abs(), p=1, dim=-1, keepdim=True) / (h * w)  # [bs, 1, 1]
            dist = nn.CosineSimilarity(dim=1)(input_pixels, norm_vec) * torch.norm(input_pixels, 2, dim=1)  # projecte distance [bs, -1]

            srt_val, _ = torch.sort(dist)  
            top_percentile = round(h * w * (1 - 0.035))
            top_dist = srt_val[:, top_percentile:top_percentile + 1]
            bottom_percentile = round(h * w * 0.035)
            bottom_dist = srt_val[:, bottom_percentile:bottom_percentile + 1]

            list_cc = []
            for bid in range(bs):
                input_img = input_pixels[bid, ...]
                idx = (dist[bid, :] >= top_dist[bid, :]) == True
                top_pixels = torch.stack([x[idx] for x in input_img])
                idx = (dist[bid, :] <= bottom_dist[bid, :]) == True
                bottom_pixels = torch.stack([x[idx] for x in input_img])

                candidates = torch.cat((top_pixels, bottom_pixels), dim=1)
                candidates = candidates - candidates.mean(dim=-1, keepdim=True).expand_as(candidates)
                _, _, V = torch.svd(candidates.t())
                if (V[:, :1]<0).sum() == 3:
                    list_cc += [-V[:, :1]]
                else:
                    list_cc += [V[:, :1]]
            list_cc = torch.stack(list_cc)

            return list_cc.view([bs, 3, 1, 1])

        elif method == 'gray':
            return torch.ones([bs, 3, 1, 1]).cuda()                                   # [-1,1]

        elif method == 'he_cc':
            wsz = 7     # 7 or 9 ?

            # proper image size for accurate local operation
            f = max(1., round(min(h, w) / 256.))
            if f > 1:
                _I = nn.functional.avg_pool2d(_I, f)

            mask = 1.

            mask_non_sat = (_I <= (1.-1./255)).cumprod(dim=1)
            mask_non_sat = mask_non_sat[:,-1:,:,:].float()
            mask *= mask_non_sat

            darkchannel, _ = torch.min(_I, dim=1, keepdim=True)
            darkchannel = 1. - nn.functional.max_pool2d(1.-darkchannel, wsz, stride=1, padding=wsz//2)

            srt_val, _ = torch.sort((darkchannel * mask_non_sat).view([bs, 1, -1]), descending=True)
            mask_dcp = (darkchannel >= srt_val[:, :, percentile: percentile + 1].unsqueeze(-1)).float()
            mask *= mask_dcp

            # topk brightness
            _I_brightness = _I.norm(p=2, dim=1, keepdim=True)
            srt_val, _ = torch.sort((_I_brightness * mask).view([bs, 1, -1]), descending=True)
            len_mask = mask.view([bs, -1]).sum(dim=-1)
            mask_bri = []
            for bid in range(bs):
                perc = int(max(1, round(len_mask[bid].item() / 100. * 10.)))
                mask_bri += [(_I_brightness[bid, ...] >= srt_val[bid, :, perc:perc + 1]).float()]
            mask_bri = torch.stack(mask_bri)
            mask *= mask_bri

            
            _A = (_I * mask).view([bs, ch, -1])
            _A = _A.sum(dim=-1) / mask.view([bs, 1, -1]).sum(dim=-1)

            return _A.view([bs, ch, 1, 1])

        else:
            raise NotImplementedError('A_method [%s] is not found' % method)


class DiscriminativeModel(nn.Module):
    def __init__(self, disc_type, config):
        super(DiscriminativeModel, self).__init__()

        if disc_type == 'rgb':
            sequence = [define_D(3, config)]
        elif disc_type == 'gray':
            sequence = [SimpleGray(), define_D(1, config)]
        elif disc_type == 'gray_input':
            sequence = [define_D(1, config)]    
        elif disc_type == 'blur':
            sequence = [GaussianBlur(sigma=3.), define_D(3, config)]
        elif disc_type == 'sat':
            sequence = [RGB2Saturation(), define_D(1, config)]
        else:
            raise NotImplementedError('discriminator [%s] is not found' % disc_type)

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class ConcatDiscriminativeModel(object):
    def __init__(self, config):
        super(ConcatDiscriminativeModel, self).__init__()
        if (config['enable'] is None) or (not config['enable']) or (len(config['enable']) == 0):
            self.enable = False
        else:
            self.enable = True
            assert len(config['enable']) == len(config['weight'])

            self.weight = config['weight']
            weight_sum = sum(config['weight'])
            assert weight_sum > 0.

            if weight_sum != 1:
                self.weight = [x / weight_sum for x in self.weight]

            self.discriminators = [DiscriminativeModel(x, config[x]) for x in config['enable']]
            self.criterion = GANLoss(use_lsgan=config['_global_settings']['use_lsgan'])
            if torch.cuda.is_available():
                self.discriminators = [nn.DataParallel(x).cuda() for x in self.discriminators]
                self.criterion = self.criterion.cuda()

    def parameters(self):
        l_params = []
        for x in self.discriminators:
            l_params += list(x.parameters())
        return l_params

    def load_state_dict(self, list_dict):
        for ith, state_dict in enumerate(list_dict):
            self.discriminators[ith].load_state_dict(state_dict)

    def state_dict(self):
        return [x.state_dict() for x in self.discriminators]

    def train(self):
        for x in self.discriminators:
            x.train()

    def eval(self):
        for x in self.discriminators:
            x.eval()

    def gan_loss(self, x, true_or_false, reduce=True, weight=1.0):
        if weight == 0.:
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(())
        else:
            l_loss = []
            for i, disc in enumerate(self.discriminators):
                l_loss += [self.weight[i] * self.criterion(disc(x), true_or_false)]
            return weight * sum(l_loss) if reduce else l_loss

    def dis_out(self, x, true_or_false, reduce=True, weight=1.0):
        if weight == 0.:
            for i, disc in enumerate(self.discriminators):
                if i == 0:
                    disc_sum = disc(x)
                else:
                    disc_sum = disc_sum+disc(x)

            discs = disc_sum / (i+1)            
            return torch.zeros(()).cuda() if torch.cuda.is_available() else torch.zeros(()), discs
        else:
            l_loss = []
            for i, disc in enumerate(self.discriminators):
                l_loss += [self.weight[i] * self.criterion(disc(x), true_or_false)]
                if i == 0:
                    disc_sum = disc(x)
                else:
                    disc_sum = disc_sum+disc(x)

        discs = disc_sum / (i+1)  
            # print(l_loss[-1].item())
        return weight * sum(l_loss) if reduce else l_loss, discs         

    def no_grad(self):
        for disc in self.discriminators:
            for param in disc.parameters():
                param.requires_grad = False

class SiameseModel(nn.Module):
    def __init__(self, n_layer = 6, inter_nc = 64, siamese_nc = 1000):
        super(SiameseModel, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.n_layer = n_layer
        model = []

        input_nc = 3

        nc_in = input_nc
        nc_out = inter_nc

        for i in range(self.n_layer):
            downconv = nn.Conv2d(nc_in, nc_out, kernel_size=4,
                                 stride=2, padding=1, bias=use_bias)
            downrelu = nn.LeakyReLU(0.2, True)
            downnorm = norm_layer(nc_out)

            model += [downconv]
            model += [downnorm]
            model += [downrelu]

            nc_in  = nc_out
            nc_out = min(nc_out*2, inter_nc*8)

        self.model = nn.Sequential(*model)

        self.fc = nn.Sequential(*[nn.Linear(8192,siamese_nc)])


    def forward(self, x):
        x_out = self.model(x)
        x_flatten = x_out.view(x_out.size(0),-1)
        out = self.fc(x_flatten)
        return out

def siamese_lossfn(logits, labels=None, diff=False, diffmargin=10., samemargin=0.):
    if diff:
        loss =  diffmargin - torch.sum(logits, dim=-1)
        loss[loss<0] = 0
        return loss
    return torch.sum(logits, axis=-1) 

class FeedbackModel(nn.Module):
    def __init__(self, inter_nc = 64):
        super(FeedbackModel, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.Conv2d(4, inter_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model1 += [nn.LeakyReLU(0.2, True)]
        model1 += [norm_layer(inter_nc)]

        model1 += [nn.Conv2d(inter_nc, inter_nc*2, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model1 += [nn.LeakyReLU(0.2, True)]
        model1 += [norm_layer(inter_nc*2)]            

        model1 += [nn.Conv2d(inter_nc*2, inter_nc*4, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model1 += [nn.LeakyReLU(0.2, True)]
        model1 += [norm_layer(inter_nc*4)]   

        model1 += [nn.Conv2d(inter_nc*4, inter_nc*8, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model1 += [nn.LeakyReLU(0.2, True)]
        model1 += [norm_layer(inter_nc*8)]               

        self.model1 = nn.Sequential(*model1)

        model2 = [nn.Conv2d(inter_nc*16, inter_nc*8, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model2 += [nn.LeakyReLU(0.2, True)]
        model2 += [norm_layer(inter_nc*8)]   

        model2 += [nn.Conv2d(inter_nc*8, inter_nc*8, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model2 += [nn.LeakyReLU(0.2, True)]
        model2 += [norm_layer(inter_nc*8)]    

        model2 += [nn.Conv2d(inter_nc*8, inter_nc*8, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model2 += [nn.LeakyReLU(0.2, True)]
        model2 += [norm_layer(inter_nc*8)]     
        
        model2 += [nn.Conv2d(inter_nc*8, inter_nc*8, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model2 += [nn.LeakyReLU(0.2, True)]
        model2 += [norm_layer(inter_nc*8)]                       

        self.model2 = nn.Sequential(*model2)

    def forward(self, img, dis, feat0):

        img_rs = F.interpolate(img, size=(128,128), mode='bilinear')
        dis_rs = F.interpolate(dis, size=(128,128), mode='bilinear')

        x = self.model1(torch.cat([img_rs,dis_rs],1))

        feat_out = self.model2(torch.cat([x,feat0],1))
        return feat_out           

class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=6, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model       
        # model_global = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=False, n_blocks=9).model  
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev  
              
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)                     