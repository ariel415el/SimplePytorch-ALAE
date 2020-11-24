from pathlib import Path
import os


def find_latest_checkpoint(ckpt_dir):
    """
    Finds the latest created .pt file in the directory
    """
    if not os.path.exists(ckpt_dir):
        return "None"
    oldest_to_newest_paths = sorted(Path(ckpt_dir).iterdir(), key=os.path.getmtime)[::-1]
    ckpts = [x._str for x in oldest_to_newest_paths if x._str.endswith("pt")]
    return ckpts[0] if ckpts else None


def get_config_str(config):
    result = f"z-{config['z_dim']}_w-{config['w_dim']}"
    if 'dataset_name' in config:
        result += f"_d-{config['dataset_name']}"

    if 'resolutions' in config:
        result += "_prog"
        for r,c in zip(config['resolutions'], config['channels']):
            result += f"-({r},{c})"

    return result