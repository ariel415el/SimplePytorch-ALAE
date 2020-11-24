import os
import torch
from datasets import get_dataset, get_dataloader
from dnn.models.ALAE import MLP_ALAE
from utils.common_utils import get_config_str
import argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='Train arguments')
parser.add_argument("--output_root", type=str, default="Training_dir-test")
parser.add_argument("--num_debug_images", type=int, default=24)
parser.add_argument("--print_model", action='store_true', default=False)
parser.add_argument("--print_config", action='store_true', default=False)
parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0/cpu")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

config = {'z_dim':50,
          'w_dim':50,
          'mapping_layers': 6,
          'image_dim':28,
          'start_lr': 0.002,
          "batch_size": 128,
          'epochs':100}

if __name__ == '__main__':
    config_descriptor = get_config_str(config)
    output_dir = os.path.join(args.output_root, f"MlpALAE_d-Mnist_{config_descriptor}")
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    # create_dataset
    train_dataset, test_dataset = get_dataset("data", "Mnist", dim=config['image_dim'])

    if args.print_config:
        print("Model config:")
        pprint(config)

    # Create model
    model = MLP_ALAE(model_config=config, device=device)
    if args.print_model:
        print(model)

    test_dataloader = get_dataloader(test_dataset, batch_size=args.num_debug_images, resize=None, device=device)
    test_samples_z = torch.randn(args.num_debug_images, config['z_dim'], dtype=torch.float32).to(device)
    test_samples = next(iter(test_dataloader))

    model.train(train_dataset, (test_samples_z, test_samples), output_dir)