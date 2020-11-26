import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from dnn.models.modules.StyleGanGenerator import StylleGanGenerator, MappingFromLatent
from dnn.models.modules.PGGanDiscriminator import PGGanDiscriminator
from dnn.custom_adam import LREQAdam
from utils.tracker import LossTracker
from dnn.costume_layers import compute_r1_gradient_penalty
from datasets import get_dataloader, EndlessDataloader


class StyleGan:
    """
    Implementation of Style Gan https://arxiv.org/pdf/1812.04948.pdf.
    Uses a style generator depicted in the paper and Uses the the discriminator and the proggressive training method
    (and other tricks) introduced in the PGGans paper https://arxiv.org/abs/1710.10196
    """
    def __init__(self,model_config, device):
        self.device = device
        self.cfg = model_config

        self.F = MappingFromLatent(num_layers=8, input_dim=self.cfg['z_dim'], out_dim=self.cfg['w_dim']).to(device).train()

        progression = list(zip(self.cfg['resolutions'], self.cfg['channels']))
        self.G = StylleGanGenerator(latent_dim=self.cfg['w_dim'], progression=progression).to(device).train()

        self.D = PGGanDiscriminator(progression=progression).to(device).train()

        self.G_optimizer = LREQAdam(list(self.F.parameters()) + list(self.G.parameters()), lr=self.cfg['lr'], betas=(0.0, 0.99), weight_decay=0)
        self.D_optimizer = LREQAdam(self.D.parameters(), lr=self.cfg['lr'], betas=(0.0, 0.99), weight_decay=0)

    def get_D_loss(self, batch_real_data, res_idx, alpha):
        batch_z = torch.randn(batch_real_data.shape[0], self.cfg['z_dim'], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            batch_fake_data = self.G(self.F(batch_z), final_resolution_idx=res_idx, alpha=alpha)
        fake_images_dicriminator_outputs = self.D(batch_fake_data, final_resolution_idx=res_idx, alpha=alpha)
        real_images_dicriminator_outputs = self.D(batch_real_data, final_resolution_idx=res_idx, alpha=alpha)
        loss = F.softplus(fake_images_dicriminator_outputs) + F.softplus(-real_images_dicriminator_outputs)
        loss = loss.reshape(-1)

        # Todo Check if r1 penalty is the default one in Style gan and verify the coeff value
        r1_penalty = compute_r1_gradient_penalty(real_images_dicriminator_outputs, batch_real_data)
        loss += self.cfg['g_penalty_coeff'] * r1_penalty
        loss = loss.mean()

        return loss

    def get_G_loss(self, batch_real_data, res_idx, alpha):
        batch_z = torch.randn(batch_real_data.shape[0], self.cfg['z_dim'], dtype=torch.float32).to(self.device)
        batch_fake_data = self.G(self.F(batch_z), final_resolution_idx=res_idx, alpha=alpha)
        fake_images_dicriminator_outputs = self.D(batch_fake_data, final_resolution_idx=res_idx, alpha=alpha)
        loss = F.softplus(-fake_images_dicriminator_outputs).mean()
        return loss

    def save_train_state(self, save_path):
        torch.save(
            {
                'F': self.F.state_dict(),
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
            },
            save_path
        )

    def set_optimizers_lr(self, new_lr):
        for optimizer in [self.D_optimizer, self.G_optimizer]:
            for group in optimizer.param_groups:
                group['lr'] = new_lr

    def train(self, train_dataset, test_data, output_dir):
        tracker = LossTracker(output_dir)
        global_steps = 0
        for res_idx, res in enumerate(self.cfg['resolutions']):
            self.set_optimizers_lr(self.cfg['learning_rates'][res_idx])
            batchs_in_phase = self.cfg['phase_lengths'][res_idx] // self.cfg['batch_sizes'][res_idx]
            dataloader = EndlessDataloader(get_dataloader(train_dataset, self.cfg['batch_sizes'][res_idx], resize=res, device=self.device))
            progress_bar = tqdm(range(batchs_in_phase * 2))
            for i in progress_bar:
                alpha = min(1.0, i / batchs_in_phase)  # < 1 in the first half and 1 in the second
                progress_bar.set_description(f"gs-{global_steps}_res-{res_idx}={res}x{res}_alpha-{alpha:.3f}")
                batch_real_data = dataloader.next()

                # train discriminator
                self.D_optimizer.zero_grad()
                loss_d = self.get_D_loss(batch_real_data, res_idx, alpha)
                loss_d.backward()
                self.D_optimizer.step()
                tracker.update(dict(loss_d=loss_d))

                if (1+i) % self.cfg['n_critic'] == 0:
                    # train generator
                    self.G_optimizer.zero_grad()
                    loss_g = self.get_G_loss(batch_real_data, res_idx, alpha)
                    loss_g.backward()
                    self.G_optimizer.step()
                    tracker.update(dict(loss_g=loss_g))
                global_steps += 1
                if global_steps % self.cfg['dump_imgs_freq'] == 0:
                    self.save_sample(global_steps, tracker, test_data, output_dir, res_idx, alpha)
            self.save_train_state(os.path.join(output_dir, 'checkpoints', f"ckpt_res-{res_idx}={res}x{res}-end.pt"))

    def save_sample(self, gs, tracker, samples_z, output_dir, res_idx, alpha):
        with torch.no_grad():
            generated_images = self.G(self.F(samples_z), res_idx, alpha)
            generated_images = torch.nn.functional.interpolate(generated_images, size=self.cfg['resolutions'][-1])

            generated_images = generated_images * 0.5 + 0.5

            tracker.register_means(gs)
            tracker.plot()
            f = os.path.join(output_dir, f"gs-{gs}_res-{self.cfg['resolutions'][res_idx]}x{self.cfg['resolutions'][res_idx]}_alpha-{alpha:.3f}.jpg")
            save_image(generated_images, f, nrow=int(np.sqrt(len(samples_z))))