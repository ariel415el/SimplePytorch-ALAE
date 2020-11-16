import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import init


def upscale_2d(x):
    return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


def downscale_2d(x):
    # since the factor is same for height and width the bilinear downsampling is just like an avg pool
    return F.avg_pool2d(x, 2, 2)  # downscales in factor 2


class LearnablePreScaleBlur(nn.Module):
    """
    From StyleFan paper:
    "We replace the nearest-neighbor up/downsampling in both net-works with bilinear sampling, which we implement by
    low-pass filtering the activations with a separable 2nd order bi-nomial filter after each upsampling layer and
    before each downsampling layer [Making Convolutional Networks Shift-Invariant Again]."
    """
    def __init__(self, channels):
        super(LearnablePreScaleBlur, self).__init__()
        f = np.array([[1, 2, 1]])
        f = np.matmul(f.transpose(), f)
        f = f / np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)
        return x


def pixel_norm(x, epsilon=1e-8):
    return x / torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)
    # return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


class StyleAffineTransform(nn.Module):
    '''
    The 'A' unit in StyleGan:
    Learned affine transform A, this module is used to transform the midiate vector w into a style vector
    it outputs a mean and std for each channel for a given number of channels.
    tis mean and std are used in AdaIn as facror and bias to chift the norm of the input
    It have n_channel

    '''
    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = LREQ_FC_Layer(dim_latent, n_channel * 2)
        # "the biases associated with mean that we initialize to one"
        self.transform.bias.data[:n_channel] = 1
        self.transform.bias.data[n_channel:] = 0

    def forward(self, w):
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


class NoiseScaler(nn.Module):
    '''
    Learned per-channel scale factor, used to scale the noise
    'B' unit in SttleGan
    '''

    def __init__(self, n_channel):
        super().__init__()
        # per channel wiehgt
        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))

    def forward(self, noise):
        result = noise * self.weight
        return result


class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    Shift the mean and variance of an input image by a given factor and bias
    '''

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1) # split number of chenels to two
        normed_input = self.norm(image)
        rescaled_image = normed_input * factor + bias
        return rescaled_image


class LREQ_FC_Layer_2(nn.Module):
    def __init__(self, in_features, out_features, lrmul=1.0):
        super(LREQ_FC_Layer_2, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.weight.data.normal_()
        self.bias.data.zero_()
        self.c = np.sqrt(2.0) / np.sqrt(self.in_features)

    def forward(self, input):
        self.weight.data *= self.c
        return F.linear(input, self.weight, self.bias)


class LREQ_FC_Layer(nn.Module):
    """
    This is a equlized learning rate version of a linear unit.
    It initializes weights with N(0,1).
    The weights are supposed to be divided by a constant each forward but this implementation is rather different.
    It must be coupled with the a costume optimizer that mutliplies the step size with the saved new attribute 'lr_equalization_coef'.
    For more information see "PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY,AND VARIATION"

    """
    def __init__(self, in_features, out_features, bias=True, gain=np.sqrt(2.0), lrmul=1.0):
        super(LREQ_FC_Layer, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.std = 0
        self.gain = gain
        self.lrmul = lrmul
        self.reset_parameters()

    def reset_parameters(self):
        self.std = self.gain / np.sqrt(self.in_features) * self.lrmul

        init.normal_(self.weight, mean=0, std=self.std / self.lrmul)
        setattr(self.weight, 'lr_equalization_coef', self.std)
        if self.bias is not None:
            setattr(self.bias, 'lr_equalization_coef', self.lrmul)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class Lreq_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, output_padding=0, dilation=1, bias=True, lrmul=1.0):
        super(Lreq_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.output_padding = (output_padding, output_padding)
        self.dilation = (dilation, dilation)
        self.lrmul = lrmul
        self.fan_in = np.prod(self.kernel_size) * in_channels

        self.weight = Parameter(torch.Tensor(out_channels, in_channels , *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.std = np.sqrt(2.0) / np.sqrt(self.fan_in)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight, mean=0, std=self.std / self.lrmul)
        setattr(self.weight, 'lr_equalization_coef', self.std)
        if self.bias is not None:
            setattr(self.bias, 'lr_equalization_coef', self.lrmul)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
