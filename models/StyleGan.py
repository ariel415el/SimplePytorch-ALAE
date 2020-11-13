from models.common_modules import *
from utils.custom_adam import LREQAdam
from utils.tracker import LossTracker
from utils.loss_utils import compute_r1_gradient_penalty
from torchvision.utils import save_image
import os
from datasets import get_dataloader
from tqdm import tqdm


RESOLUTIONS = [4, 8, 16, 32, 64]
LEARNING_RATES = [0.001, 0.001, 0.001, 0.001, 0.001, ]
TRAIN_PHASE_LENGTH = 128 * 30
BATCH_SIZES = [128, 128, 128, 128, 128]
N_CRITIC=1


class EndlessDataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            real_image = next(self.iterator)

        except (OSError, StopIteration):
            self.iterator = iter(self.dataloader)
            real_image = next(self.iterator)

        return real_image

class StyleGan:
    def __init__(self, z_dim, w_dim, image_dim, hyper_parameters, device):
        self.device = device
        self.z_dim = z_dim
        self.hp = {'lr': 0.002, 'g_penalty_coeff':10.0}
        self.hp.update(hyper_parameters)

        self.F = MappingFromLatent(num_layers=8, input_dim=z_dim, out_dim=w_dim).to(device).train()

        self.G = StylleGanGenerator(latent_dim=w_dim, out_dim=image_dim).to(device).train()

        self.D = PGGanDiscriminator().to(device).train()

        self.G_optimizer = LREQAdam(list(self.F.parameters()) + list(self.G.parameters()), lr=self.hp['lr'], betas=(0.0, 0.99), weight_decay=0)
        self.D_optimizer = LREQAdam(self.D.parameters(), lr=self.hp['lr'], betas=(0.0, 0.99), weight_decay=0)

    def get_D_loss(self, batch_real_data, res_idx, alpha):
        batch_z = torch.randn(batch_real_data.shape[0], self.z_dim, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            batch_fake_data = self.G(self.F(batch_z), final_resolution_idx=res_idx, alpha=alpha)
        fake_images_dicriminator_outputs = self.D(batch_fake_data, res_idx, alpha)
        real_images_dicriminator_outputs = self.D(batch_real_data, res_idx, alpha)
        loss = F.softplus(fake_images_dicriminator_outputs) + F.softplus(-real_images_dicriminator_outputs)
        loss = loss.reshape(-1)
        r1_penalty = compute_r1_gradient_penalty(real_images_dicriminator_outputs, batch_real_data)

        loss += self.hp['g_penalty_coeff'] * r1_penalty
        loss = loss.mean()

        return loss

    def get_G_loss(self, batch_real_data, res_idx, alpha):
        batch_z = torch.randn(batch_real_data.shape[0], self.z_dim, dtype=torch.float32).to(self.device)
        batch_fake_data = self.G(self.F(batch_z), final_resolution_idx=res_idx, alpha=alpha)
        fake_images_dicriminator_outputs = self.D(batch_fake_data, final_resolution_idx=res_idx, alpha=alpha)
        loss = F.softplus(-fake_images_dicriminator_outputs).mean()
        return loss

    def train(self, train_dataset, test_data, output_dir):
        tracker = LossTracker(output_dir)
        global_steps = 0
        progress_bar = tqdm(enumerate(RESOLUTIONS))
        for res_idx, res in progress_bar:
            # TODO adjust optimizers learning rate
            batchs_in_phase = TRAIN_PHASE_LENGTH // BATCH_SIZES[res_idx]
            dataloader = EndlessDataloader(get_dataloader(train_dataset, BATCH_SIZES[res_idx], resize=res, device=self.device))
            for i in range(batchs_in_phase * 2) :
                alpha = min(1, i / batchs_in_phase) # < 1 in the first half and 1 in the second
                progress_bar.set_description(f"gs-{global_steps}_res-{res}x{res}_alpha-{alpha:.3f}")
                batch_real_data = dataloader.next()

                # train discriminator
                self.D_optimizer.zero_grad()
                loss_d = self.get_D_loss(batch_real_data, res_idx, alpha)
                loss_d.backward()
                self.D_optimizer.step()
                tracker.update(dict(loss_d=loss_d))

                if (1+i) % N_CRITIC == 0:
                    # train generator
                    self.G_optimizer.zero_grad()
                    loss_g = self.get_G_loss(batch_real_data, res_idx, alpha)
                    loss_g.backward()
                    self.G_optimizer.step()
                    tracker.update(dict(loss_g=loss_g))
                global_steps += 1

                if global_steps % 500 == 1:
                    self.save_sample( global_steps, tracker, test_data[1], test_data[0], output_dir, res_idx, alpha)


    def save_sample(self, gs, tracker, samples, samples_z, output_dir, res_idx, alpha):
        with torch.no_grad():
            generated_images = self.G(self.F(samples_z), res_idx, alpha)
            generated_images = torch.nn.functional.interpolate(generated_images, size=samples.shape[-1])

            generated_images = generated_images * 0.5 + 0.5

            tracker.register_means(gs)
            tracker.plot()
            f = os.path.join(output_dir, f"gs-{gs}_res-{RESOLUTIONS[res_idx]}x{RESOLUTIONS[res_idx]}_alpha-{alpha}.jpg")
            save_image(generated_images, f, nrow=len(samples))