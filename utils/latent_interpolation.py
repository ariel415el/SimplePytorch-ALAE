import matplotlib.pyplot as plt
import numpy as np

def plot_latent_interpolation(model, start_images, end_images, steps=5, plot_path="latnet_interpolation.png"):
    N = start_images.shape[0]
    C, H, W = start_images.shape[1:]
    canvas = np.zeros((C, H * (steps + 1), W * N))

    start_latents = model.encode(start_images)
    end_latents = model.encode(end_images)
    yticks = ['start img']
    for i in range(steps + 1):
        alpha = i / steps
        yticks += [f"recon\nalpha={alpha}"]
        interpolated_latents = (1-alpha) * start_latents + alpha * end_latents
        interpolated_images = model.decode(interpolated_latents).detach().numpy()

        row = np.concatenate(interpolated_images, axis=2)* 0.5 + 0.5

        canvas[:, i * H: (i+1) * H, :] = row
    yticks += ['end img']

    start_row = np.concatenate(start_images.detach().numpy(), axis=2) * 0.5 + 0.5
    end_row = np.concatenate(end_images.detach().numpy(), axis=2)* 0.5 + 0.5
    canvas = np.concatenate([start_row, canvas, end_row], axis=1)
    plt.imshow(canvas.transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
    plt.yticks(np.arange(H/2, canvas.shape[1], H) , yticks)
    plt.savefig(plot_path)
    plt.clf()