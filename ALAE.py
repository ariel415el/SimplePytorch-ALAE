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
        self.latent_size = latent_size

        self.D = DiscriminatorFC(
            w_dim=latent_size,
            mapping_layers=3)

        self.F = VAEMappingFromLatent(
            z_dim=latent_size,
            w_dim=latent_size,
            mapping_layers=mapping_layers)

        self.G = GeneratorFC(latent_size=latent_size)

        self.E = EncoderFC(latent_size=latent_size)


    def generate(self, z, return_w=False):

        w_vector = self.F(z)

        image = self.G.forward(w_vector)
        if return_w:
            return w_vector, image
        else:
            return image

    def encode(self, imgs):
        w_vecs = self.E(imgs)
        critic_scores = self.D(w_vecs)
        return w_vecs, critic_scores

    def forward(self, batch_images, d_train, ae):
        batch_z = torch.randn(batch_images.shape[0], self.latent_size, dtype=torch.float32).to(self.device)
        if ae:
            W_from_z, fake_img = self.generate(z=batch_z, return_w=True)

            self.E.requires_grad_(True)
            W_from_img, _ = self.encode(fake_img)

            assert W_from_z.shape == W_from_img.shape

            W_reconstruction_loss = torch.mean(((W_from_img - W_from_z.detach())**2))

            return W_reconstruction_loss

        elif d_train:
            with torch.no_grad():
                fake_images = self.generate(z=batch_z)
            self.E.requires_grad_(True)

            _, d_result_fake = self.encode(fake_images.detach())
            _, d_result_real = self.encode(batch_images)

            loss_d = discriminator_logistic_simple_gp(d_result_fake, d_result_real, batch_images)
            return loss_d
        else:
            self.E.requires_grad_(False)
            fake_images = self.generate(z=batch_z.detach())

            _, d_result_fake = self.encode(fake_images)

            loss_g = generator_logistic_non_saturating(d_result_fake)

            return loss_g
