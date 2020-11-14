import os
import imageio
from pathlib import Path


def make_gif(directory, duration_per_image=0.5, limit_images=20):
    images = []
    oldest_to_newest_files = sorted(Path(directory).iterdir(), key=os.path.getmtime)
    for filename in oldest_to_newest_files:
        fpath = filename._str
        if (fpath.endswith('.jpg') or fpath.endswith('.png')) and "plot.png" not in fpath:
            images.append(imageio.imread(fpath))
    images = images[::len(images)//limit_images]
    imageio.mimsave(os.path.join(os.path.dirname(directory), os.path.basename(directory) + ".gif"), images, duration=duration_per_image)

if __name__ == '__main__':
    make_gif('Training_dir/StyleGAN_LFW/now_tricks')