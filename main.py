import torch
from ALAE import ALAE
from datasets import get_dataset, get_dataloader
import numpy as np

OUTPUT_DIR= 'training_dir_3_opts'
LATENT_SPACE_SIZE = 50
NUMED_BUG_IMAGES=32
EPOCHS=100
DATASET="mnist"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def trainGAN():
    # Create datasets
    train_dataset, test_dataset, img_dim = get_dataset("../data", DATASET)

    # Create model
    hyper_parameters = {'lr': 0.002, "batch_size": 128, 'mapping_layers':6}
    model = ALAE(LATENT_SPACE_SIZE, device, hyper_parameters, image_dim=img_dim)

    dataloader = get_dataloader(train_dataset, hyper_parameters['batch_size'], device=device)
    test_dataloader = get_dataloader(test_dataset, batch_size=NUMED_BUG_IMAGES, device=device)
    test_samples_z = torch.tensor(np.random.RandomState(3456).randn(NUMED_BUG_IMAGES, LATENT_SPACE_SIZE),
                                  dtype=torch.float32).to(device)
    test_samples = next(iter(test_dataloader))

    model.train(dataloader, (test_samples_z, test_samples), EPOCHS, OUTPUT_DIR)


if __name__ == '__main__':
    trainGAN()