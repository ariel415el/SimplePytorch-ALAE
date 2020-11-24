import os
import torch
from datasets import get_dataset, get_dataloader
from dnn.models.ALAE import StyleALAE
from utils.common_utils import find_latest_checkpoint, get_config_str
import argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='Train arguments')
parser.add_argument("--output_root", type=str, default="Training_dir-test")
parser.add_argument("--dataset_name", type=str, default="FFHQ", help='FFHQ/CelebA/LFW')
parser.add_argument("--num_debug_images", type=int, default=32)
parser.add_argument("--print_model", action='store_true', default=False)
parser.add_argument("--print_config", action='store_true', default=False)
parser.add_argument("--device", type=str, default="cpu", help="cuda:0/cpu")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

config = {
    "z_dim": 512,
    "w_dim": 512,
    "image_dim": 64,
    "mapping_layers": 8,
    "resolutions": [4, 8, 16, 32, 64, 64, 64],
    "channels": [256, 256, 128, 128, 64, 32, 16],
    "learning_rates": [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.003, 0.003],
    "phase_lengths": [200_000, 400_000, 600_000, 800_000, 1000_000, 1000_000, 1000_000],
    "batch_sizes": [256, 256, 128, 64, 32, 32, 32],
    "n_critic": 1,
    "dump_imgs_freq": 1000,
    "checkpoint_freq": 1
}

if __name__ == '__main__':
    config_descriptor = get_config_str(config)

    output_dir = os.path.join(args.output_root, f"StyleALAE-{config_descriptor}")
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    cfg_file_path = os.path.join(output_dir, 'checkpoints', "cfg.pt_pkl")
    if os.path.exists(cfg_file_path):
        print("Overriding model config from file...")
        config = torch.load(cfg_file_path)
    torch.save(config, cfg_file_path)

    if args.print_config:
        print("Model config:")
        pprint(config)

    # create_dataset
    train_dataset, test_dataset = get_dataset("data", args.dataset_name, dim=config['resolutions'][-1])

    # Create model
    model = StyleALAE(model_config=config, device=device)
    model.load_train_state(find_latest_checkpoint(os.path.join(output_dir, 'checkpoints')))
    if args.print_model:
        print(model)

    test_dataloader = get_dataloader(test_dataset, batch_size=args.num_debug_images, resize=None, device=device)
    test_samples_z = torch.randn(args.num_debug_images, config['z_dim'], dtype=torch.float32).to(device)
    test_samples = next(iter(test_dataloader))

    model.train(train_dataset, (test_samples_z, test_samples), output_dir)