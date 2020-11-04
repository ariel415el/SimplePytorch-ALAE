import torch
from ALAE_new import ALAE
from dataloader import get_dataloader
import numpy as np

OUTPUT_DIR= 'training_dir_3_opts'
LATENT_SPACE_SIZE = 50
NUMED_BUG_IMAGES=32
EPOCHS=100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def trainGAN():
    # Create model
    hyper_parameters = {'lr': 0.002, "batch_size": 128, 'mapping_layers':6}
    model = ALAE(LATENT_SPACE_SIZE, device, hyper_parameters)

    # Create datasets
    dataloader = get_dataloader(hyper_parameters['batch_size'], device=device)
    test_dataloader = get_dataloader(batch_size=NUMED_BUG_IMAGES, device=device, test_set=True)
    test_samples_z = torch.tensor(np.random.RandomState(3456).randn(NUMED_BUG_IMAGES, LATENT_SPACE_SIZE),
                                  dtype=torch.float32).to(device)
    test_samples = next(iter(test_dataloader))

    model.train(dataloader, (test_samples_z, test_samples), EPOCHS, OUTPUT_DIR)

if __name__ == '__main__':
    trainGAN()