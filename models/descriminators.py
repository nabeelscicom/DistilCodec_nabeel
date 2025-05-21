import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm

from utils import get_padding
from utils import init_weights

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1))), weight_norm(
                        Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            padding=get_padding(kernel_size, 1))), weight_norm(
                                Conv1d(
                                    channels,
                                    channels,
                                    kernel_size,
                                    1,
                                    dilation=1,
                                    padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3,
                 use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(
                Conv2d(
                    1,
                    32, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    32,
                    128, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    128,
                    512, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    512,
                    1024, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, mpd_config: dict = None):
        super(MultiPeriodDiscriminator, self).__init__()
        
        self.mpd_config = None
        
        if not mpd_config:
            self.discriminators = nn.ModuleList([
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ])
        else:
            self.mpd_config = mpd_config
            self.discriminators = nn.ModuleList(
                [DiscriminatorP(period=period,
                                kernel_size=self.mpd_config['kernal_size'],
                                stride=self.mpd_config['stride']) 
                 for period in self.mpd_config['periods']]
            )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, 
                 use_spectral_norm=False,
                 kernal_sizes=None,
                 strides=None,
                 paddings=None):
        
        super(DiscriminatorS, self).__init__()
        
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        
        if not (kernal_sizes and strides and paddings):
            self.convs = nn.ModuleList([
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ])
        else:
            self.convs = nn.ModuleList([
                norm_f(Conv1d(1, 128, kernal_sizes[0], strides[0], padding=paddings[0])),
                norm_f(Conv1d(128, 128, kernal_sizes[1], strides[1], groups=4, padding=paddings[1])),
                norm_f(Conv1d(128, 256, kernal_sizes[2], strides[2], groups=16, padding=paddings[2])),
                norm_f(Conv1d(256, 512, kernal_sizes[3], strides[3], groups=16, padding=paddings[3])),
                norm_f(Conv1d(512, 1024, kernal_sizes[4], strides[4], groups=16, padding=paddings[4])),
                norm_f(Conv1d(1024, 1024, kernal_sizes[5], strides[5], groups=16, padding=paddings[5])),
                norm_f(Conv1d(1024, 1024, kernal_sizes[6], strides[6], padding=paddings[6])),
            ])
            
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, msd_config: dict = None):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.msd_config = msd_config
        
        if self.msd_config is not None:
            self.discriminators = nn.ModuleList([
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ])
            self.meanpools = nn.ModuleList(
                [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])
        else:
            discriminator_config = self.msd_config['DiscriminatorS']
            kernal_sizes = discriminator_config['kernal_sizes']
            strides = discriminator_config['strides']
            paddings = discriminator_config['paddings']
            self.discriminators = nn.ModuleList(
                [DiscriminatorS(use_spectral_norm=True if i == 0 else False,
                                kernal_sizes=kernal_sizes,
                                strides=strides,
                                paddings=paddings)
                 for i in range(3)]
            )
            
            avg_pool_config = self.msd_config['avg_poolings']
            collated_ap_config = zip(avg_pool_config['kernal_sizes'],
                                     avg_pool_config['stridess'],
                                     avg_pool_config['paddings'])
            self.meanpools = nn.ModuleList(
                [AvgPool1d(kernel_size=config[0], 
                           stride=config[1], 
                           padding=config[2])
                 for config in collated_ap_config]
            )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses



