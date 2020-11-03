from models import *
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



class Model(nn.Module):
    def __init__(self, layer_count, latent_size, mapping_layers, channels, device):
        super(Model, self).__init__()
        self.layer_count = layer_count
        self.device = device

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

        self.latent_size = latent_size
        self.style_mixing_prob = 0.9

    def generate(self, z, return_styles=False):

        styles = self.mapping_fl(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_fl.num_layers, 1)

        rec = self.decoder.forward(styles)
        if return_styles:
            return s, rec
        else:
            return rec

    def encode(self, x):
        Z = self.encoder(x)
        Z_ = self.mapping_tl(Z)
        return Z[:,None, :], Z_[:, 0]

    def forward(self, x, d_train, ae):
        z = torch.randn(x.shape[0], self.latent_size, dtype=torch.float32).to(self.device)
        if ae:
            self.encoder.requires_grad_(True)

            s, rec = self.generate(z=z, return_styles=True)

            Z, _ = self.encode(rec)

            assert Z.shape == s.shape

            Lae = torch.mean(((Z - s.detach())**2))

            return Lae

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(z=z)

            self.encoder.requires_grad_(True)

            _, d_result_real = self.encode(x)

            _, d_result_fake = self.encode(Xp.detach())

            loss_d = discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)
            return loss_d
        else:
            self.encoder.requires_grad_(False)

            rec = self.generate(z=z.detach())

            _, d_result_fake = self.encode(rec)

            loss_g = generator_logistic_non_saturating(d_result_fake)

            return loss_g

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_tl.parameters()) + list(self.mapping_fl.parameters()) \
                     + list(self.decoder.parameters()) + list(self.encoder.parameters())
            other_param = list(other.mapping_tl.parameters()) + list(other.mapping_fl.parameters()) \
                          + list(other.decoder.parameters()) + list(other.encoder.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)