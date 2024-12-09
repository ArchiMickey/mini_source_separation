import torch
from tqdm.auto import tqdm



@torch.no_grad()
def sample_discrete_euler(model, x, steps, sigma_max=1, verbose=True, **extra_args):
    """Draws samples from a model given starting noise. Euler method"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)

    #alphas, sigmas = 1-t, t
    for t_curr, t_prev in tqdm(zip(t[:-1], t[1:]), disable=not verbose):
            # Broadcast the current timestep to the correct shape
            t_curr_tensor = t_curr * torch.ones(
                (x.shape[0],), dtype=x.dtype, device=x.device
            )
            dt = t_prev - t_curr  # we solve backwards in our formulation
            x = x + dt * model(x, t_curr_tensor, **extra_args) #.denoise(x, denoiser, t_curr_tensor, cond, uc)

    # If we are on the last timestep, output the denoised image
    return x