from dnn.costume_layers import *

class PGGanDescriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, is_last_block=False):
        super(PGGanDescriminatorBlock, self).__init__()
        assert not (is_last_block and downsample), "You should not downscale after last block"
        self.downsample = downsample
        self.is_last_block = is_last_block
        self.conv1 = Lreq_Conv2d(in_channels, out_channels, 3, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        if downsample:
            self.conv2 = torch.nn.Sequential(LearnablePreScaleBlur(out_channels),
                                             Lreq_Conv2d(out_channels, out_channels, 3, 1),
                                             torch.nn.AvgPool2d(2, 2))
        elif is_last_block:
            self.conv2 = Lreq_Conv2d(out_channels, out_channels, STARTING_DIM, 0)
        else:
            self.conv2 = Lreq_Conv2d(out_channels, out_channels, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.lrelu(x)

        return x


class PGGanDiscriminator(nn.Module):
    def __init__(self, progression):
        super().__init__()

        self.from_rgbs = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Parse the module description given in "progression"
        for i in range(len(progression)):
            self.from_rgbs.append(Lreq_Conv2d(3, progression[i][1], 1, 0))
        for i in range(len(progression) - 1, -1, -1):
            if i == 0:
                self.conv_blocks.append(PGGanDescriminatorBlock(progression[i][1] + 1, STARTING_CHANNELS,
                                                              is_last_block=True))
            else:
                downsample = progression[i][0] / 2 == progression[i - 1][0]
                self.conv_blocks.append(PGGanDescriminatorBlock(progression[i][1], progression[i - 1][1],
                                                              downsample=downsample))


        assert(len(self.conv_blocks) == len(self.from_rgbs))
        self.fc = LREQ_FC_Layer(512, 1)
        self.n_layers = len(self.conv_blocks)

    def forward(self, image, final_resolution_idx, alpha=1):
        feature_maps = self.from_rgbs[final_resolution_idx](image)

        first_layer_idx = self.n_layers - final_resolution_idx - 1
        for i in range(first_layer_idx, self.n_layers):
            # Before final layer, do minibatch stddev:  adds a constant std channel
            if i == self.n_layers - 1:
                res_var = feature_maps.var(0, unbiased=False) + 1e-8 # Avoid zero
                res_std = torch.sqrt(res_var)
                mean_std = res_std.mean().expand(feature_maps.size(0), 1, STARTING_DIM, STARTING_DIM)
                feature_maps = torch.cat([feature_maps, mean_std], 1)

            feature_maps = self.conv_blocks[i](feature_maps)

            # If this is the first conv block to be run and this is not the last one the there is an already stabilized
            # previous scale layers : Alpha blend the output of the unstable new layer with the downscaled putput
            # of the previous one
            if i == first_layer_idx and i != self.n_layers - 1 and alpha < 1:
                if self.conv_blocks[i].downsample:
                    image = downscale_2d(image)
                skip_first_block_feature_maps = self.from_rgbs[final_resolution_idx - 1](image)
                feature_maps = alpha * feature_maps + (1 - alpha) * skip_first_block_feature_maps

        # Convert it into [batch, channel(512)], so the fully-connetced layer
        # could process it.
        feature_maps = feature_maps.squeeze(2).squeeze(2)
        result = self.fc(feature_maps)
        return result
