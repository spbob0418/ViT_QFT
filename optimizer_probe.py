import torch
import math

def compute_adamw_update(exp_avg, exp_avg_sq, step, lr, betas, eps):
    """
    Computes the update term for the AdamW optimizer.

    Parameters:
    - exp_avg (Tensor): Exponential moving average of the gradients.
    - exp_avg_sq (Tensor): Exponential moving average of the squared gradients.
    - step (int): The current step number.

    Returns:
    - Tensor: The update term to be added to the parameters.
    """
    beta1, beta2 = betas

    # Compute bias corrections
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    # Compute the denominator
    denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add(eps)

    # Compute the step size
    step_size = lr / bias_correction1

    # Compute the update term
    adamw_update = exp_avg.div(denom).mul(-step_size)

    # print(adamw_update.shape)
    # print(type(adamw_update.abs().mean()))
    # adamw_update.abs().mean().item()
    scalar_mean = adamw_update.abs().mean().item()


    return scalar_mean
