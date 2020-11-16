from models.common_modules import *
from utils.custom_adam import LREQAdam
from tqdm import tqdm
from torchvision.utils import save_image
import os
from utils.tracker import LossTracker
from utils.loss_utils import compute_r1_gradient_penalty
from datasets import get_dataloader, EndlessDataloader


class ALAE:
    def __init__(self, z_dim, w_dim, image_dim, architecture_mode, hyper_parameters, device):
        self.device = device
        self.z_dim = z_dim
        self.hp = {'lr': 0.002, 'mapping_layers': 6, "g_penalty_coeff": 10, 'descriminator_layers':3}
        self.hp.update(hyper_parameters)

        if architecture_mode == "FC":
            self.G = GeneratorFC(latent_dim=w_dim, output_img_dim=image_dim).to(device).train()
            self.E = EncoderFC(input_img_dim=image_dim, latent_dim=w_dim).to(device).train()
        else:
            self.G = StylleGanGenerator(latent_dim=w_dim, output_img_dim=image_dim).to(device).train()
            self.E = AlaeEncoder(input_img_dim=image_dim, latent_dim=w_dim).to(device).train()
        self.F = MappingFromLatent(input_dim=z_dim, out_dim=w_dim, num_layers=self.hp['mapping_layers']).to(device).train()
        self.D = DiscriminatorFC(input_dim=w_dim, num_layers=self.hp['descriminator_layers']).to(device).train()

        self.ED_optimizer = LREQAdam(list(self.E.parameters()) + list(self.D.parameters()), lr=self.hp['lr'], betas=(0.0, 0.99), weight_decay=0)
        self.FG_optimizer = LREQAdam(list(self.F.parameters()) + list(self.G.parameters()), lr=self.hp['lr'], betas=(0.0, 0.99), weight_decay=0)
        # self.EG_optimizer = LREQAdam(list(self.E.parameters()) + list(self.G.parameters()), lr=self.hp['lr'], betas=(0.0, 0.99), weight_decay=0)

    def get_ED_loss(self, batch_real_data, **ae_kwargs):
        """
        Computes a standard adverserial loss for the dictriminator D(E( * )):
          how much  D(E( * )) can differentiate between real images and images generated by G(F( * ))
         """
        batch_z = torch.randn(batch_real_data.shape[0], self.z_dim, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            batch_fake_data = self.G(self.F(batch_z), **ae_kwargs)
        fake_images_dicriminator_outputs = self.D(self.E(batch_fake_data, **ae_kwargs))
        real_images_dicriminator_outputs = self.D(self.E(batch_real_data, **ae_kwargs))
        loss = F.softplus(fake_images_dicriminator_outputs) + F.softplus(-real_images_dicriminator_outputs)

        r1_penalty = compute_r1_gradient_penalty(real_images_dicriminator_outputs, batch_real_data)

        loss += self.hp['g_penalty_coeff'] * r1_penalty
        loss = loss.mean()

        return loss

    def get_FG_loss(self, batch_real_data, **ae_kwargs):
        """
        Computes a standard adverserial loss for the generator:
            how much  G(F( * )) can fool D(E ( * ))
        """
        batch_z = torch.randn(batch_real_data.shape[0], self.z_dim, dtype=torch.float32).to(self.device)
        batch_fake_data = self.G(self.F(batch_z), **ae_kwargs)
        fake_images_dicriminator_outputs = self.D(self.E(batch_fake_data, **ae_kwargs))
        loss = F.softplus(-fake_images_dicriminator_outputs).mean()

        return loss

    def get_EG_loss(self, batch_real_data, **ae_kwargs):
        """
        Compute a reconstruction loss in the w latent space for the auto encoder (E,G):
            || F(X) - E(G(F(x))) || = || W - E(G(W)) ||
        """
        batch_z = torch.randn(batch_real_data.shape[0], self.z_dim, dtype=torch.float32).to(self.device)
        batch_w = self.F(batch_z)
        batch_reconstructed_w = self.E(self.G(batch_w, **ae_kwargs), **ae_kwargs)
        return torch.mean(((batch_reconstructed_w - batch_w.detach())**2))

    def perform_train_step(self, batch_real_data, tracker, **ae_kwargs):
        """
        Optimizes the model with a batch of real images:
             optimize :Disctriminator, Generator and reconstruction loss of the autoencoder
        """
        # Step I. Update E, and D: optimizer the discriminator D(E( * ))
        self.ED_optimizer.zero_grad()
        L_adv_ED = self.get_ED_loss(batch_real_data, **ae_kwargs)
        L_adv_ED.backward()
        self.ED_optimizer.step()
        tracker.update(dict(L_adv_ED=L_adv_ED))

        # Step II. Update F, and G: Optimize the generator G(F( * )) to fool D(E ( * ))
        self.FG_optimizer.zero_grad()
        L_adv_FG = self.get_FG_loss(batch_real_data, **ae_kwargs)
        L_adv_FG.backward()
        self.FG_optimizer.step()
        tracker.update(dict(L_adv_FG=L_adv_FG))

        # Step III. Update E, and G: Optimize the reconstruction loss in the Latent space W
        self.ED_optimizer.zero_grad()
        self.FG_optimizer.zero_grad()
        # self.EG_optimizer.zero_grad()
        L_err_EG = self.get_EG_loss(batch_real_data, **ae_kwargs)
        L_err_EG.backward()
        # self.EG_optimizer.step()
        self.ED_optimizer.step()
        self.FG_optimizer.step()
        tracker.update(dict(L_err_EG=L_err_EG))

    def train(self, train_dataset, test_data, output_dir):
        raise NotImplementedError

    def generate(self, z_vectors, **ae_kwargs):
        raise NotImplementedError

    def encode(self, img, **ae_kwargs):
        raise NotImplementedError

    def decode(self, latent_vectorsz, **ae_kwargs):
        raise NotImplementedError

    def save_sample(self, dump_path, samples_z, samples, **ae_kwargs):
        with torch.no_grad():
            restored_image = self.decode(self.encode(samples, **ae_kwargs), **ae_kwargs)
            generated_images = self.generate(samples_z, **ae_kwargs)

            resultsample = torch.cat([samples, restored_image, generated_images], dim=0).cpu()

            # Normalize images from -1,1 to 0, 1.
            # Eventhough train samples are in this range (-1,1), the generated image may not. But this should diminish as
            # raining continues or else the discriminator can detect them. Anyway save_image clamps it to 0,1
            resultsample = resultsample * 0.5 + 0.5

            save_image(resultsample, dump_path, nrow=len(samples))

class StyleALAE(ALAE):
    def __init__(self, z_dim, w_dim, image_dim, hyper_parameters, device):
        super().__init__( z_dim, w_dim, image_dim, 'cnn', hyper_parameters, device)
        self.hp.update({
            "resolutions": [4, 8, 16, 32, 64],
            "learning_rates": [0.001, 0.001, 0.001, 0.001, 0.001],
            "phase_lengths": [400_000, 600_000, 800_000, 1_000_000, 2_000_000],
            "batch_sizes": [128, 128, 128, 128, 128],
            "n_critic": 1,
            "dump_imgs_freq" : 1000
                   })
        self.hp.update(hyper_parameters)

    def set_optimizers_lr(self, new_lr):
        for optimizer in [self.ED_optimizer, self.FG_optimizer]:
            for group in optimizer.param_groups:
                group['lr'] = new_lr

    def generate(self, z_vectors, **ae_kwargs):
        self.G.eval()
        self.F.eval()
        generated_images = self.G(self.F(z_vectors), **ae_kwargs)
        self.G.train()
        self.F.train()
        return torch.nn.functional.interpolate(generated_images, size=self.hp['resolutions'][-1])

    def encode(self, img, **ae_kwargs):
        desired_resolution = self.hp['resolutions'][ae_kwargs['final_resolution_idx']]
        downscaled_images = torch.nn.functional.interpolate(img, size=desired_resolution)
        self.E.eval()
        w_vectors = self.E(downscaled_images, **ae_kwargs)
        self.E.train()
        return w_vectors

    def decode(self, w_vectors, **ae_kwargs):
        self.G.eval()
        generated_images = self.G(w_vectors, **ae_kwargs)
        self.G.train()
        return torch.nn.functional.interpolate(generated_images, size=self.hp['resolutions'][-1])

    def train(self, train_dataset, test_data, output_dir):
        tracker = LossTracker(output_dir)
        global_steps = 0
        for res_idx, res in enumerate(self.hp['resolutions']):
            self.set_optimizers_lr(self.hp['learning_rates'][res_idx])
            batch_size = self.hp['batch_sizes'][res_idx]
            batchs_in_phase = self.hp['phase_lengths'][res_idx] // batch_size
            dataloader = EndlessDataloader(get_dataloader(train_dataset, batch_size, resize=res, device=self.device))
            progress_bar = tqdm(range(batchs_in_phase * 2))
            for i in progress_bar:
                # first half of the batchs are fade in phase where alpha < 1. in the second half alpha =1
                alpha = min(1.0, i / batchs_in_phase)
                progress_bar.set_description(f"gs-{global_steps}_res-{res}x{res}_alpha-{alpha:.3f}")
                batch_real_data = dataloader.next()

                self.perform_train_step(batch_real_data, tracker, final_resolution_idx=res_idx, alpha=alpha)

                if global_steps % self.hp['dump_imgs_freq'] == 0:
                    tracker.register_means(global_steps)
                    tracker.plot()
                    dump_path = os.path.join(output_dir, f"gs-{global_steps}_res-{res}x{res}_alpha-{alpha}.jpg")
                    self.save_sample(dump_path, test_data[0], test_data[1], final_resolution_idx=res_idx, alpha=alpha)


class FC_ALAE(ALAE):
    def __init__(self, z_dim, w_dim, image_dim, hyper_parameters, device):
        super().__init__( z_dim, w_dim, image_dim, 'FC', hyper_parameters, device)
        self.hp.update({"batch_size": 128, "epochs":100, 'mapping_layers':8})
        self.hp.update(hyper_parameters)

    def generate(self, z_vectors, **ae_kwargs):
        return self.G(self.F(z_vectors))

    def encode(self, img, **ae_kwargs):
        return self.E(img)

    def decode(self, latent_vectors, **ae_kwargs):
        return self.D(latent_vectors)

    def train(self, train_dataset, test_data, output_dir):
        train_dataloader = get_dataloader(train_dataset, self.hp['batch_size'], resize=None, device=self.device)
        tracker = LossTracker(output_dir)
        for epoch in range(self.hp['epochs']):
            for batch_real_data in tqdm(train_dataloader):
                self.perform_train_step(batch_real_data, tracker)

            tracker.register_means(epoch)
            tracker.plot()
            dump_path = os.path.join(output_dir, f"epoch-{epoch}.jpg")
            self.save_sample(tracker, test_data[0], test_data[1])

