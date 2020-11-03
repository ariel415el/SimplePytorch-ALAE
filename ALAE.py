from models import *


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
    def __init__(self, latent_size, mapping_layers, device):
        super(Model, self).__init__()
        self.device = device

        self.discriminator = DiscriminatorFC(
            w_dim=latent_size,
            mapping_layers=3)

        self.mapping_fl = VAEMappingFromLatent(
            z_dim=latent_size,
            w_dim=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GeneratorFC(latent_size=latent_size)

        self.encoder = EncoderFC(latent_size=latent_size)

        self.latent_size = latent_size

    def generate(self, z, return_w=False):

        w_vector = self.mapping_fl(z)

        image = self.decoder.forward(w_vector)
        if return_w:
            return w_vector, image
        else:
            return image

    def encode(self, imgs):
        w_vecs = self.encoder(imgs)
        critic_scores = self.discriminator(w_vecs)
        return w_vecs, critic_scores

    def forward(self, batch_images, d_train, ae):
        batch_z = torch.randn(batch_images.shape[0], self.latent_size, dtype=torch.float32).to(self.device)
        if ae:
            W_from_z, fake_img = self.generate(z=batch_z, return_w=True)

            self.encoder.requires_grad_(True)
            W_from_img, _ = self.encode(fake_img)

            assert W_from_z.shape == W_from_img.shape

            W_reconstruction_loss = torch.mean(((W_from_img - W_from_z.detach())**2))

            return W_reconstruction_loss

        elif d_train:
            with torch.no_grad():
                fake_images = self.generate(z=batch_z)
            self.encoder.requires_grad_(True)

            _, d_result_fake = self.encode(fake_images.detach())
            _, d_result_real = self.encode(batch_images)

            loss_d = discriminator_logistic_simple_gp(d_result_fake, d_result_real, batch_images)
            return loss_d
        else:
            self.encoder.requires_grad_(False)
            fake_images = self.generate(z=batch_z.detach())

            _, d_result_fake = self.encode(fake_images)

            loss_g = generator_logistic_non_saturating(d_result_fake)

            return loss_g

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.discriminator.parameters()) + list(self.mapping_fl.parameters()) \
                     + list(self.decoder.parameters()) + list(self.encoder.parameters())
            other_param = list(other.discriminator.parameters()) + list(other.mapping_fl.parameters()) \
                          + list(other.decoder.parameters()) + list(other.encoder.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)