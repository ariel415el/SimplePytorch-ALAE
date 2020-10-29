import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import init


class MappingBlock(nn.Module):
    def __init__(self, inputs, output, lrmul):
        super(MappingBlock, self).__init__()
        self.fc = Linear(inputs, output, lrmul=lrmul)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x


def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


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
    def __init__(self, layer_count, latent_size, channels=3):
        super(EncoderFC, self).__init__()
        self.layer_count = layer_count
        self.channels = channels
        self.latent_size = latent_size

        self.fc_1 = Linear(28 * 28, 1024)
        self.fc_2 = Linear(1024, 1024)
        self.fc_3 = Linear(1024, latent_size)

    def encode(self, x, lod):
        x = F.interpolate(x, 28)
        x = x.view(x.shape[0], 28 * 28)

        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)
        x = F.leaky_relu(x, 0.2)

        return x

    def forward(self, x, lod, blend):
        return self.encode(x, lod)


class GeneratorFC(nn.Module):
    def __init__(self, layer_count=3, latent_size=128, channels=3):
        super(GeneratorFC, self).__init__()
        self.layer_count = layer_count
        self.channels = channels
        self.latent_size = latent_size

        self.fc_1 = Linear(latent_size, 1024)
        self.fc_2 = Linear(1024, 1024)
        self.fc_3 = Linear(1024, 28 * 28)

    def decode(self, x, lod, blend_factor, noise):
        if len(x.shape) == 3:
            x = x[:, 0]  # no styles
        x.view(x.shape[0], self.latent_size)

        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)

        x = x.view(x.shape[0], 1, 28, 28)
        x = F.interpolate(x, 2 ** (2 + lod))
        return x

    def forward(self, x, lod, blend_factor, noise):
        return self.decode(x, lod, blend_factor, noise)


class VAEMappingFromLatent(nn.Module):
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(VAEMappingFromLatent, self).__init__()
        inputs = dlatent_size
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = latent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)
            #print("dense %d %s" % ((i + 1), millify(count_parameters(block))))

    def forward(self, x):
        x = pixel_norm(x)

        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)

        return x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1)


class VAEMappingToLatentNoStyle(nn.Module):
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(VAEMappingToLatentNoStyle, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = Linear(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)

    def forward(self, x):
        for i in range(self.mapping_layers):
            if i == self.mapping_layers - 1:
                #x = self.map_blocks[i](x)
                x = self.map_blocks[i](x)
            else:
                #x = self.map_blocks[i](x)
                x = self.map_blocks[i](x)
        return x