from costume_layers import *


class MappingFromLatent(nn.Module):
    def __init__(self, num_layers=5, z_dim=256, w_dim=256):
        super(MappingFromLatent, self).__init__()
        layers = [LREQ_FC_Layer(z_dim, w_dim, lrmul=0.1), nn.LeakyReLU(0.2)]
        for i in range(num_layers - 1):
            layers += [LREQ_FC_Layer(w_dim, w_dim, lrmul=0.1), nn.LeakyReLU(0.2)]
        self.mapping = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = pixel_norm(x)

        x = self.mapping(x)

        return x


class DiscriminatorFC(nn.Module):
    def __init__(self, num_layers, w_dim=256):
        super(DiscriminatorFC, self).__init__()
        assert num_layers >= 2
        layers = []
        for i in range(num_layers):
            out_dim = 1 if i == num_layers - 1 else w_dim
            layers += [LREQ_FC_Layer(w_dim, out_dim, lrmul=0.1), nn.LeakyReLU(0.2)]
        self.mapping = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        x = x.view(-1)
        return x


class EncoderFC(nn.Module):
    def __init__(self, input_size, latent_size):
        super(EncoderFC, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size

        self.fc_1 = LREQ_FC_Layer(input_size ** 2, 1024)
        self.fc_2 = LREQ_FC_Layer(1024, 1024)
        self.fc_3 = LREQ_FC_Layer(1024, latent_size)

    def encode(self, x):
        x = x.view(x.shape[0], self.input_size**2)

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
    def __init__(self, latent_size, output_size):
        super(GeneratorFC, self).__init__()
        self.latent_size = latent_size
        self.output_size = output_size

        self.fc_1 = LREQ_FC_Layer(latent_size, 1024)
        self.fc_2 = LREQ_FC_Layer(1024, 1024)
        self.fc_3 = LREQ_FC_Layer(1024, self.output_size ** 2)

    def forward(self, x):
        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)

        x = x.view(x.shape[0], 1, self.output_size, self.output_size)

        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_size, noise_dim, in_channels, out_channels, is_first_block=False):
        super(GeneratorBlock, self).__init__()
        if is_first_block:
            self.conv1 = ConstantInput(in_channels, 4)
        else:
            self.conv1 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                             Lreq_Conv2d(in_channels, out_channels, 3, padding=1))

        self.style_affine_transform_1 = StyleAffineTransform(latent_size, in_channels)
        self.style_affine_transform_2 = StyleAffineTransform(latent_size, in_channels)
        self.noise_scaler_1 = NoiseScaler(in_channels)
        self.noise_scaler_2 = NoiseScaler(in_channels)
        self.adain = AdaIn(in_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = Lreq_Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, input, latent_w, noise):
        result = self.conv1(input) + self.noise_scaler_1(noise)
        result = self.adain(result, self.style_affine_transform_1(latent_w))
        result = self.lrelu(result)

        result = self.conv_2(result) + self.noise_scaler_2(noise)
        result = self.adain(result, self.style_affine_transform_2(latent_w))
        result = self.lrelu(result)

        return result

class ProgressingGenerator(nn.Module):
    def __init__(self, latent_size, noise_dim=64, out_dim=64):
        assert out_dim == 64
        super(ProgressingGenerator, self).__init__()
        self.latent_size = latent_size

        self.progression = nn.ModuleList(
            [
                GeneratorBlock(latent_size, noise_dim, 512, 512, is_first_block=True),  # 4
                GeneratorBlock(latent_size, noise_dim, 512, 512),  # 8
                GeneratorBlock(latent_size, noise_dim, 512, 512),  # 16
                GeneratorBlock(latent_size, noise_dim, 512, 512),  # 32
                GeneratorBlock(latent_size, noise_dim, 512, 256)  # 64
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                Lreq_Conv2d(512, 3, 1),
                Lreq_Conv2d(512, 3, 1),
                Lreq_Conv2d(512, 3, 1),
                Lreq_Conv2d(512, 3, 1),
                Lreq_Conv2d(256, 3, 1),
            ]
        )

    def forward(self, w, noise, final_resolution_idx, alpha):
        generated_img = None
        feature_maps_upsample = None
        for i, block in enumerate(self.progression):
            if i == 0:
                feature_maps = block(w, noise)
            else:
                feature_maps_upsample = nn.functional.interpolate(feature_maps, scale_factor=2, mode='bilinear', align_corners=False)
                feature_maps = block(feature_maps_upsample, w, noise)

            if i == final_resolution_idx:
                generated_img = self.to_rgb[i](feature_maps)

                # blend with upsampled image that havent been passed throw the last block
                if i > 0:
                    prev_scale_generated_img = self.to_rgbs[i - 1](feature_maps_upsample)
                    generated_img = alpha * generated_img + (1 - alpha) * prev_scale_generated_img

                break

        return generated_img