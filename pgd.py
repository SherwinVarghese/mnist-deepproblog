import torch.nn as nn
from typing import Callable

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def pgd(
    model,
    adv_epsilon: float,
    input: torch.Tensor,
    final_step_labels: torch.Tensor,
    final_layer=False,
    num_steps=40,
    step_size=1e-1,
    return_model_output=False,
):
    """Given an input, run a PGD attack on a specified model and epsilon. Return the
    number of successful attacks.
    """
    # TODO EdS: Hard coded for magnitude task
    x_ = input.clone()

    # attack_input_ub = torch.min(x_ + adv_epsilon, torch.ones(x_.shape))
    # attack_input_lb = torch.max(x_ - adv_epsilon, torch.zeros(x_.shape))
    attack_input_ub = x_ + adv_epsilon
    attack_input_lb = x_ - adv_epsilon

    adv_data = pgd_attack(
        model,
        attack_input_lb,
        attack_input_ub,
        lambda logits: nn.CrossEntropyLoss()(logits, final_step_labels),
        num_steps,
        step_size,
    )

    adv_model_output = model(adv_data)
    # The model performs softmax in the last layer
    # adv_model_output = torch.nn.Softmax(dim=1)(adv_model_output)

    if return_model_output:
        # results, adv_output, adv_inpt
        # return torch.max(adv_model_output, dim=1).indices != torch.max(final_step_labels, dim=1).indices, adv_model_output, adv_data
        return torch.max(adv_model_output, dim=1).indices != final_step_labels, adv_model_output, adv_data

    return torch.max(adv_model_output, dim=1).indices != final_step_labels



def pgd_attack(
    model, input_lb, input_ub, loss_function: Callable, n_steps, step_size, final_layer=False,
):

    step_size_scaling = (input_ub - input_lb) / 2
    attack_point = input_lb.clone()
    attack_loss = (-np.inf) * torch.ones(
        input_lb.shape[0], dtype=torch.float32, device=input_lb.device
    )

    with torch.enable_grad():
        # Sample uniformly in input domain
        adv_input = (
            torch.zeros_like(input_lb).uniform_() * (input_ub - input_lb) + input_lb
        ).detach_()

        for i in range(n_steps):
            adv_input.requires_grad = True
            if adv_input.grad is not None:
                adv_input.grad.zero_()

            adv_outs = model(adv_input)
            if final_layer:
                adv_outs = torch.nn.Softmax()(adv_outs)  # TODO EdS: Remove final_layer flag

            adv_outs = adv_outs
            obj = loss_function(torch.squeeze(adv_outs, 0))

            attack_point = torch.where(
                (obj >= attack_loss).view((-1,) + (1,) * (input_lb.dim() - 1)),
                adv_input.detach().clone(),
                attack_point,
            )
            attack_loss = torch.where(
                obj >= attack_loss, obj.detach().clone(), attack_loss
            )

            grad = torch.autograd.grad(obj.sum(), adv_input)[0]
            adv_input = adv_input.detach() + step_size * step_size_scaling * grad.sign()
            adv_input = torch.max(torch.min(adv_input, input_ub), input_lb).detach_()

    if n_steps > 1:
        adv_outs = model(adv_input)

        if final_layer:
            adv_outs = torch.nn.Softmax()(adv_outs)  # TODO EdS: Remove final_layer flag

        obj = loss_function(torch.squeeze(adv_outs, 0))
        attack_point = torch.where(
            (obj >= attack_loss).view((-1,) + (1,) * (input_lb.dim() - 1)),
            adv_input.detach().clone(),
            attack_point,
        )
    else:
        attack_point = adv_input.detach().clone()

    return attack_point

