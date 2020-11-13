from utils.costume_layers import *


class MappingFromLatent(nn.Module):
    def __init__(self, num_layers=5, input_dim=256, out_dim=256):
        super(MappingFromLatent, self).__init__()
        layers = [LREQ_FC_Layer(input_dim, out_dim, lrmul=0.1), nn.LeakyReLU(0.2)]
        for i in range(num_layers - 1):
            layers += [LREQ_FC_Layer(out_dim, out_dim, lrmul=0.1), nn.LeakyReLU(0.2)]
        self.mapping = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = pixel_norm(x)

        x = self.mapping(x)

        return x


class DiscriminatorFC(nn.Module):
    def __init__(self, num_layers, input_dim=256):
        super(DiscriminatorFC, self).__init__()
        assert num_layers >= 2
        layers = []
        for i in range(num_layers):
            out_dim = 1 if i == num_layers - 1 else input_dim
            layers += [LREQ_FC_Layer(input_dim, out_dim, lrmul=0.1), nn.LeakyReLU(0.2)]
        self.mapping = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        x = x.view(-1)
        return x


class EncoderFC(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(EncoderFC, self).__init__()
        self.out_dim = out_dim
        self.input_dim = input_dim

        self.fc_1 = LREQ_FC_Layer(input_dim ** 2, 1024)
        self.fc_2 = LREQ_FC_Layer(1024, 1024)
        self.fc_3 = LREQ_FC_Layer(1024, out_dim)

    def encode(self, x):
        x = x.view(x.shape[0], self.input_dim**2)

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
    def __init__(self, input_dim, output_dim):
        super(GeneratorFC, self).__init__()
        self.latent_size = input_dim
        self.output_dim = output_dim

        self.fc_1 = LREQ_FC_Layer(input_dim, 1024)
        self.fc_2 = LREQ_FC_Layer(1024, 1024)
        self.fc_3 = LREQ_FC_Layer(1024, self.output_dim ** 2)

    def forward(self, x):
        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)

        x = x.view(x.shape[0], 1, self.output_dim, self.output_dim)

        return x


class StyleGeneratorBlock(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, is_first_block=False):
        super(StyleGeneratorBlock, self).__init__()
        if is_first_block:
            self.conv1 = ConstantInput(out_channels, 4)
        else:
            self.conv1 = Lreq_Conv2d(in_channels, out_channels, 3, padding=1)

        self.style_affine_transform_1 = StyleAffineTransform(latent_dim, out_channels)
        self.style_affine_transform_2 = StyleAffineTransform(latent_dim, out_channels)
        self.noise_scaler_1 = NoiseScaler(out_channels)
        self.noise_scaler_2 = NoiseScaler(out_channels)
        self.adain = AdaIn(in_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = Lreq_Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, input, latent_w, noise):
        result = self.conv1(input) + self.noise_scaler_1(noise)
        result = self.adain(result, self.style_affine_transform_1(latent_w))
        result = self.lrelu(result)

        result = self.conv2(result) + self.noise_scaler_2(noise)
        result = self.adain(result, self.style_affine_transform_2(latent_w))
        result = self.lrelu(result)

        return result


class StylleGanGenerator(nn.Module):
    def __init__(self, latent_dim, out_dim=64):
        assert out_dim == 64
        super(StylleGanGenerator, self).__init__()
        self.latent_dim = latent_dim

        self.progression = nn.ModuleList(
            [
                StyleGeneratorBlock(latent_dim, 512, 256, is_first_block=True),  # 4x4 img
                StyleGeneratorBlock(latent_dim, 256, 128),  # 8x8 img
                StyleGeneratorBlock(latent_dim, 128, 64),  # 16x16 img
                StyleGeneratorBlock(latent_dim, 64, 32),  # 32x32 img
                StyleGeneratorBlock(latent_dim, 32, 16)  # 64x64 img
            ]
        )
        self.to_rgb = nn.ModuleList(
            [
                Lreq_Conv2d(256, 3, 1, 0),
                Lreq_Conv2d(128, 3, 1, 0),
                Lreq_Conv2d(64, 3, 1, 0),
                Lreq_Conv2d(32, 3, 1, 0),
                Lreq_Conv2d(16, 3, 1, 0),
            ]
        )

    def forward(self, w, final_resolution_idx, alpha):
        generated_img = None
        feature_maps_upsample = None
        feature_maps = None
        noise = torch.randn((w.shape[0], 1, 1, 1), dtype=torch.float32).to(w.device)
        for i, block in enumerate(self.progression):
            if i == 0:
                feature_maps = block(w, w, noise) # TODO solve the issue where thi needs an input
            else:
                feature_maps_upsample = nn.functional.interpolate(feature_maps, scale_factor=2, mode='bilinear', align_corners=False)
                feature_maps = block(feature_maps_upsample, w, noise)

            if i == final_resolution_idx:
                generated_img = self.to_rgb[i](feature_maps)

                # If there is an already stabilized last previous resolution layer. alpha blend with it
                if i > 0 and alpha < 1:
                    generated_img_without_last_block = self.to_rgbs[i - 1](feature_maps_upsample)
                    generated_img = alpha * generated_img + (1 - alpha) * generated_img_without_last_block
                break

        return generated_img


class PGGanDiscriminator(nn.Module):
    def __init__(self):
        self.resolutions = [4,8,16,32,64]
        super().__init__()
        # Waiting to adjust the size
        self.from_rgbs = nn.ModuleList([
            Lreq_Conv2d(3, 256, 1, 0),  # 4x4 imgs
            Lreq_Conv2d(3, 128, 1, 0),
            Lreq_Conv2d(3, 64, 1, 0),
            Lreq_Conv2d(3, 32, 1, 0),
            Lreq_Conv2d(3, 16, 1, 0) # 64x64 imgs
        ])
        self.convs  = nn.ModuleList([
            nn.Sequential(Lreq_Conv2d(16, 32, 3, 1), nn.LeakyReLU(0.2), Lreq_Conv2d(32, 32, 3, 1), nn.LeakyReLU(0.2)), # 64x64 images start from here
            nn.Sequential(Lreq_Conv2d(32, 64, 3, 1), nn.LeakyReLU(0.2), Lreq_Conv2d(64, 64, 3, 1), nn.LeakyReLU(0.2)), # 32x32 images start from here
            nn.Sequential(Lreq_Conv2d(64, 128, 3, 1), nn.LeakyReLU(0.2), Lreq_Conv2d(128, 128, 3, 1), nn.LeakyReLU(0.2)), # 16x16 images start from here
            nn.Sequential(Lreq_Conv2d(128, 256, 3, 1), nn.LeakyReLU(0.2), Lreq_Conv2d(256, 256, 3, 1), nn.LeakyReLU(0.2)), # 8x8 images start from here
            nn.Sequential(Lreq_Conv2d(256 + 1, 512, 3, 1), nn.LeakyReLU(0.2), Lreq_Conv2d(512, 512, 4, 0), nn.LeakyReLU(0.2)), # 4x4 images start from here
        ])
        assert(len(self.convs) == len(self.from_rgbs))
        self.fc = LREQ_FC_Layer(512, 1)
        self.n_layers = len(self.convs)


    def forward(self, image, final_resolution_idx, alpha=1):
        feature_maps = self.from_rgbs[final_resolution_idx](image)
        # If there is an already stabilized previous scale layers: Alpha Fade in new discriminator layers
        if final_resolution_idx > 0 and alpha < 1:
            down_sampled_image = nn.functional.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
            feature_maps = alpha * feature_maps + (1 - alpha) * self.from_rgbs[final_resolution_idx - 1](down_sampled_image)
        for i in range(self.n_layers - final_resolution_idx - 1, self.n_layers):
            # Before final layer, do minibatch stddev:  adds a constant std channel
            if i == self.n_layers - 1:
                res_var = feature_maps.var(0, unbiased=False) + 1e-8 # Avoid zero
                res_std = torch.sqrt(res_var)
                mean_std = res_std.mean().expand(feature_maps.size(0), 1, 4, 4)
                feature_maps = torch.cat([feature_maps, mean_std], 1)

            # Conv
            feature_maps = self.convs[i](feature_maps)

            # Not the final layer
            if i != self.n_layers - 1:
                # Downsample for further usage
                feature_maps = nn.functional.interpolate(feature_maps, scale_factor=0.5, mode='bilinear',
                                                                     align_corners=False)

        # Convert it into [batch, channel(512)], so the fully-connetced layer
        # could process it.
        feature_maps = feature_maps.squeeze(2).squeeze(2)
        result = self.fc(feature_maps)
        return result