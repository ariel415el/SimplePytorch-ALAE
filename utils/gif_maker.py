import os
import imageio
from pathlib import Path


def make_gif(directory, duration_per_image=0.7, limit_images=40):
    images = []
    # oldest_to_newest_files = sorted(Path(directory).iterdir(), key=os.path.getmtime)
    fnames = sorted(os.listdir(directory), key=lambda x: int(x.split('_')[0][3:]))
    for filename in fnames:
        # fpath = filename.as_posix()
        fpath = os.path.join(directory, filename)
        if (fpath.endswith('.jpg') or fpath.endswith('.png')) and "plot.png" not in fpath:
            img = imageio.imread(fpath)
            images.append(img[:,:img.shape[1]*2//3])
            # images.append(img)
    # images = images[::len(images)//limit_images]
    imageio.mimsave(os.path.join(os.path.dirname(directory), os.path.basename(directory) + ".gif"), images, duration=duration_per_image)

if __name__ == '__main__':
    # make_gif('Training_dir/StyleGAN_LFW/now_tricks')
    make_gif('images_')