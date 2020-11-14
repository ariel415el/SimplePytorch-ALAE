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
        self.is_first_block = is_first_block

        if is_first_block:
            self.const_input = nn.Parameter(torch.randn(1, out_channels, 4, 4))
        else:
            self.blur = LearnablePreScaleBlur(out_channels)
            self.conv1 = Lreq_Conv2d(in_channels, out_channels, 3, padding=1)

        self.style_affine_transform_1 = StyleAffineTransform(latent_dim, out_channels)
        self.style_affine_transform_2 = StyleAffineTransform(latent_dim, out_channels)
        self.noise_scaler_1 = NoiseScaler(out_channels)
        self.noise_scaler_2 = NoiseScaler(out_channels)
        self.adain = AdaIn(in_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = Lreq_Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, input, latent_w, noise):
        if self.is_first_block:
            assert(input is None)
            result = self.const_input.repeat(latent_w.shape[0], 1, 1, 1)
        else:
            result = upscale_2d(input)
            result = self.conv1(result)
            result = self.blur(result)

        result += self.noise_scaler_1(noise)
        result = self.adain(result, self.style_affine_transform_1(latent_w))
        result = self.lrelu(result)

        result = self.conv2(result)
        result += self.noise_scaler_2(noise)
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
        feature_maps = None
        for i, block in enumerate(self.progression):
            # Separate noise for each block
            noise = torch.randn((w.shape[0], 1, 1, 1), dtype=torch.float32).to(w.device)

            prev_feature_maps = feature_maps
            feature_maps = block(feature_maps, w, noise)

            if i == final_resolution_idx:
                generated_img = self.to_rgb[i](feature_maps)

                # If there is an already stabilized last previous resolution layer. alpha blend with it
                if i > 0 and alpha < 1:
                    generated_img_without_last_block = upscale_2d(self.to_rgb[i - 1](prev_feature_maps))

                    generated_img = alpha * generated_img + (1 - alpha) * generated_img_without_last_block
                break

        return generated_img


class PGGanDescriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k1, p1, k2, p2, downsample=True):
        super(PGGanDescriminatorBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = Lreq_Conv2d(in_channels, out_channels, k1, p1)
        self.lrelu = nn.LeakyReLU(0.2)
        if downsample:
            self.conv2 = torch.nn.Sequential(LearnablePreScaleBlur(out_channels),
                                             Lreq_Conv2d(out_channels, out_channels, k2, p2),
                                             torch.nn.AvgPool2d(2, 2))
        else:
            self.conv2 = Lreq_Conv2d(out_channels, out_channels, k2, p2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.lrelu(x)

        return x

class PGGanDiscriminator(nn.Module):
    def __init__(self):
        # self.resolutions = [4,8,16,32,64]
        super().__init__()
        # Waiting to adjust the size
        self.from_rgbs = nn.ModuleList([
            Lreq_Conv2d(3, 256, 1, 0),  # 4x4 imgs
            Lreq_Conv2d(3, 128, 1, 0),
            Lreq_Conv2d(3, 64, 1, 0),
            Lreq_Conv2d(3, 32, 1, 0),
            Lreq_Conv2d(3, 16, 1, 0) # 64x64 imgs
        ])
        self.convs = nn.ModuleList([
            PGGanDescriminatorBlock(16, 32, k1=3, p1=1, k2=3, p2=1),
            PGGanDescriminatorBlock(32, 64, k1=3, p1=1, k2=3, p2=1),
            PGGanDescriminatorBlock(64, 128, k1=3, p1=1, k2=3, p2=1),
            PGGanDescriminatorBlock(128, 256, k1=3, p1=1, k2=3, p2=1),
            PGGanDescriminatorBlock(256 + 1, 512, k1=3, p1=1, k2=4, p2=0, downsample=False)
        ])
        assert(len(self.convs) == len(self.from_rgbs))
        self.fc = LREQ_FC_Layer(512, 1)
        self.n_layers = len(self.convs)

    def forward(self, image, final_resolution_idx, alpha=1):
        feature_maps = self.from_rgbs[final_resolution_idx](image)
        # If there is an already stabilized previous scale layers: Alpha Fade in new discriminator layers
        first_layer_idx = self.n_layers - final_resolution_idx - 1
        for i in range(first_layer_idx, self.n_layers):
            # Before final layer, do minibatch stddev:  adds a constant std channel
            if i == self.n_layers - 1:
                res_var = feature_maps.var(0, unbiased=False) + 1e-8 # Avoid zero
                res_std = torch.sqrt(res_var)
                mean_std = res_std.mean().expand(feature_maps.size(0), 1, 4, 4)
                feature_maps = torch.cat([feature_maps, mean_std], 1)

            # Conv
            feature_maps = self.convs[i](feature_maps)

            # If there is an already stabilized previous scale layers (not last layer):
            # Alpha blend the output of the unstable new layer with the downscaled putput of the previous one
            if i == first_layer_idx and i != self.n_layers - 1 and alpha < 1:
                down_sampled_image = downscale_2d(image)
                feature_maps = alpha * feature_maps + (1 - alpha) * self.from_rgbs[final_resolution_idx - 1](
                    down_sampled_image)

        # Convert it into [batch, channel(512)], so the fully-connetced layer
        # could process it.
        feature_maps = feature_maps.squeeze(2).squeeze(2)
        result = self.fc(feature_maps)
        return result