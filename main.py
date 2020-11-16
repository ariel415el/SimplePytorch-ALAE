from datasets import get_dataset, get_dataloader
from models.StyleGan import StyleGan
from models.ALAE import *

OUTPUT_DIR = 'Training_dir-test'
NUMED_BUG_IMAGES=36
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_ALAE_mnist(output_dir):
    LATENT_SPACE_SIZE= 50
    EPOCHS = 100
    hp = {'lr': 0.002, "batch_size": 128, 'mapping_layers':6, 'epochs':EPOCHS}
    train_dataset, test_dataset, img_dim = get_dataset("data", "Mnist")

    # Create model
    model = FC_ALAE(z_dim=LATENT_SPACE_SIZE, w_dim=LATENT_SPACE_SIZE, image_dim=img_dim, hyper_parameters=hp, device=device)

    test_dataloader = get_dataloader(test_dataset, batch_size=NUMED_BUG_IMAGES, resize=None, device=device)
    test_samples_z = torch.randn(NUMED_BUG_IMAGES, LATENT_SPACE_SIZE, dtype=torch.float32).to(device)
    test_samples = next(iter(test_dataloader))

    model.train(train_dataset, (test_samples_z, test_samples), output_dir)


def train_StyleGan_lfw(output_dir):

    # Create datasets
    LATENT_SPACE_SIZE = 512
    train_dataset, test_dataset, img_dim = get_dataset("data", 'lfw')

    # Create model
    model = StyleGan(z_dim=LATENT_SPACE_SIZE, w_dim=LATENT_SPACE_SIZE, image_dim=img_dim, hyper_parameters={},device=device)

    test_samples_z = torch.randn(NUMED_BUG_IMAGES, LATENT_SPACE_SIZE, dtype=torch.float32).to(device)

    model.train(train_dataset, test_samples_z, output_dir)


def train_ALAE_lfw(output_dir):

    # Create datasets
    LATENT_SPACE_SIZE = 512
    train_dataset, test_dataset, img_dim = get_dataset("data", 'lfw')
    hp = {
            "resolutions": [4, 8, 16, 32, 64],
            "learning_rates": [0.001, 0.001, 0.001, 0.001, 0.001],
            # "phase_lengths": [400_000, 600_000, 800_000, 1_000_000, 2_000_000],
            "phase_lengths": [128, 128, 128, 128, 128],
            "batch_sizes": [128, 128, 128, 128, 128],
            "n_critic": 1
                   }
    # Create model
    model = StyleALAE(z_dim=LATENT_SPACE_SIZE, w_dim=LATENT_SPACE_SIZE, image_dim=img_dim, hyper_parameters=hp,device=device)

    test_dataloader = get_dataloader(test_dataset, batch_size=NUMED_BUG_IMAGES, resize=None, device=device)
    test_samples_z = torch.randn(NUMED_BUG_IMAGES, LATENT_SPACE_SIZE, dtype=torch.float32).to(device)
    test_samples = next(iter(test_dataloader))

    model.train(train_dataset, (test_samples_z, test_samples), output_dir)


if __name__ == '__main__':
    # train_ALAE_mnist(OUTPUT_DIR + "/ALAE_mnist")
    # train_StyleGan_lfw(OUTPUT_DIR + "/StyleGAN_LFW")
    train_ALAE_lfw(OUTPUT_DIR + "/StyleALAE_LFW")