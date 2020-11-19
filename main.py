from datasets import get_dataset, get_dataloader
from models.StyleGan import StyleGan
from models.ALAE import *
from pathlib import Path

OUTPUT_DIR = 'Training_dir-test'
NUMED_BUG_IMAGES=36
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return "None"
    oldest_to_newest_paths = sorted(Path(ckpt_dir).iterdir(), key=os.path.getmtime)
    return [x._str for x in oldest_to_newest_paths if x._str.endswith("pt")][0]


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


def train_StyleALAE_on_faces(output_dir, dataset_name):
    # Create datasets
    output_dir  = os.path.join(output_dir, f"StyleALAE-{dataset_name}")
    LATENT_SPACE_SIZE = 512
    dim = 32
    train_dataset, test_dataset = get_dataset("data", dataset_name, dim=dim)
    hp = {
            "resolutions": [4, 8, 16, 32, 32, 32, 32],
            "channels": [512, 512, 256, 128, 64, 32, 16],
            "learning_rates": [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.003, 0.003],
            "phase_lengths": [200_000, 400_000, 600_000, 800_000, 1000_000, 1000_000, 1000_000],
            "batch_sizes": [512, 256, 128, 64, 64, 64, 64],
            "n_critic": 1,
            "dump_imgs_freq": 2000,
            "checkpoint_freq": 10000
                   }

    # # debug code
    # hp["phase_lengths"] = [8] * len(hp["resolutions"])
    # hp["batch_sizes"] = [8] * len(hp["resolutions"])

    # Create model
    model = StyleALAE(z_dim=LATENT_SPACE_SIZE, w_dim=LATENT_SPACE_SIZE, image_dim=dim, hyper_parameters=hp, device=device)
    model.load_train_state(find_latest_checkpoint(os.path.join(output_dir, 'checkpoints')))

    test_dataloader = get_dataloader(test_dataset, batch_size=NUMED_BUG_IMAGES, resize=None, device=device)
    test_samples_z = torch.randn(NUMED_BUG_IMAGES, LATENT_SPACE_SIZE, dtype=torch.float32).to(device)
    test_samples = next(iter(test_dataloader))

    model.train(train_dataset, (test_samples_z, test_samples), output_dir)


if __name__ == '__main__':
    # train_ALAE_mnist(OUTPUT_DIR + "/ALAE_mnist")
    # train_StyleGan_lfw(OUTPUT_DIR + "/StyleGAN_LFW")
    train_StyleALAE_on_faces(OUTPUT_DIR, 'LFW')