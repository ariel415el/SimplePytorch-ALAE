import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import init


def pixel_norm(x, epsilon=1e-8):
    return x / torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, gain=np.sqrt(2.0), lrmul=1.0):
        super(Linear, self).__init__()
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


class EncoderFC(nn.Module):
    def __init__(self, latent_size):
        super(EncoderFC, self).__init__()
        self.latent_size = latent_size

        self.fc_1 = Linear(28 * 28, 1024)
        self.fc_2 = Linear(1024, 1024)
        self.fc_3 = Linear(1024, latent_size)

    def encode(self, x):
        x = x.view(x.shape[0], 28 * 28)

        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)
        x = F.leaky_relu(x, 0.2)

        return x

    def forward(self, x):
        return self.encode(x)


class GeneratorFC(nn.Module):
    def __init__(self, latent_size=128):
        super(GeneratorFC, self).__init__()
        self.latent_size = latent_size

        self.fc_1 = Linear(latent_size, 1024)
        self.fc_2 = Linear(1024, 1024)
        self.fc_3 = Linear(1024, 28 * 28)

    def decode(self, x):
        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)

        x = x.view(x.shape[0], 1, 28, 28)
        return x

    def forward(self, x):
        return self.decode(x)


class VAEMappingFromLatent(nn.Module):
    def __init__(self, mapping_layers=5, z_dim=256, w_dim=256):
        super(VAEMappingFromLatent, self).__init__()
        layers = [Linear(z_dim, w_dim, lrmul=0.1)]
        for i in range(mapping_layers - 1):
            layers += [Linear(w_dim, w_dim, lrmul=0.1), nn.LeakyReLU(0.2)]
        self.mapping = torch.nn.Sequential(*layers)
    def forward(self, x):
        x = pixel_norm(x)

        x = self.mapping(x)

        return x


class DiscriminatorFC(nn.Module):
    def __init__(self, mapping_layers, w_dim=256):
        super(DiscriminatorFC, self).__init__()
        assert mapping_layers >= 2
        layers = []
        for i in range(mapping_layers):
            out_dim = 1 if i == mapping_layers - 1 else w_dim
            layers += [Linear(w_dim, out_dim, lrmul=0.1), nn.LeakyReLU(0.2)]
        self.mapping = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        x = x.view(-1)
        return x