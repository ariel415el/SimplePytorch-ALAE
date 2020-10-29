from reproduce_ALAE.models import *
import random

def discriminator_logistic_simple_gp(d_result_fake, d_result_real, reals, r1_gamma=10.0):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real))

    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
        loss = loss + r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def generator_logistic_non_saturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()


class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)

class Model(nn.Module):
    def __init__(self, layer_count, latent_size, mapping_layers, channels):
        super(Model, self).__init__()
        self.layer_count = layer_count

        self.mapping_tl = VAEMappingToLatentNoStyle(
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=3)

        self.mapping_fl = VAEMappingFromLatent(
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GeneratorFC(
            layer_count=layer_count,
            latent_size=latent_size,
            channels=channels)

        self.encoder = EncoderFC(
            layer_count=layer_count,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg_beta = 0.995
        self.dlatent_avg = DLatent(latent_size, self.mapping_fl.num_layers)
        self.latent_size = latent_size
        self.style_mixing_prob = 0.9

    def generate(self, lod, blend_factor, z=None, count=32, mixing=True, noise=True, return_styles=False, no_truncation=False):
        if z is None:
            z = torch.randn(count, self.latent_size)
        styles = self.mapping_fl(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_fl.num_layers, 1)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

        if mixing and self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                styles2 = self.mapping_fl(z2)[:, 0]
                styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(1, self.mapping_fl.num_layers, 1)

                layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)


        rec = self.decoder.forward(styles, lod, blend_factor, noise)
        if return_styles:
            return s, rec
        else:
            return rec

    def encode(self, x, lod, blend_factor):
        Z = self.encoder(x, lod, blend_factor)
        Z_ = self.mapping_tl(Z)
        return Z[:,None, :], Z_[:, 0]

    def forward(self, x, lod, blend_factor, d_train, ae):
        if ae:
            self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.latent_size)
            s, rec = self.generate(lod, blend_factor, z=z, mixing=False, noise=True, return_styles=True)

            Z, d_result_real = self.encode(rec, lod, blend_factor)

            assert Z.shape == s.shape

            Lae = torch.mean(((Z - s.detach())**2))

            return Lae

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(lod, blend_factor, count=x.shape[0], noise=True)

            self.encoder.requires_grad_(True)

            _, d_result_real = self.encode(x, lod, blend_factor)

            _, d_result_fake = self.encode(Xp.detach(), lod, blend_factor)

            loss_d = discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)
            return loss_d
        else:
            with torch.no_grad():
                z = torch.randn(x.shape[0], self.latent_size)

            self.encoder.requires_grad_(False)

            rec = self.generate(lod, blend_factor, count=x.shape[0], z=z.detach(), noise=True)

            _, d_result_fake = self.encode(rec, lod, blend_factor)

            loss_g = generator_logistic_non_saturating(d_result_fake)

            return loss_g

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_tl.parameters()) + list(self.mapping_fl.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_tl.parameters()) + list(other.mapping_fl.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)