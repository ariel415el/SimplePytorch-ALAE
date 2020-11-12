import torch


def compute_r1_gradient_penalty(d_result_real, real_images):
    real_grads = torch.autograd.grad(d_result_real.sum(), real_images, create_graph=True, retain_graph=True)[0]
    # real_grads = torch.autograd.grad(d_result_real, real_images,
    #                                  grad_outputs=torch.ones_like(d_result_real),
    #                                  create_graph=True, retain_graph=True)[0]
    r1_penalty = 0.5 * torch.sum(real_grads.pow(2.0), dim=[1, 2, 3]) # Norm on all dims but batch

    return r1_penalty
