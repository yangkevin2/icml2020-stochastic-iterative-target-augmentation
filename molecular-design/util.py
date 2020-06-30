from argparse import Namespace
import os
from typing import Dict, List, Tuple

from scipy.spatial.distance import cosine as cosine_distance
import numpy as np
import torch
import torch.nn as nn

from dataset import PAD_TOKEN, START_TOKEN, END_TOKEN
from model import Model


def pad_batch(batch: np.ndarray, pad_index: int = 0, min_length: int = 0) -> Tuple[torch.LongTensor, torch.LongTensor]:
    lengths = [len(s) for s in batch]
    max_len = max(lengths)
    max_len = max(max_len, min_length)
    batch = [np.pad(s, (0, max_len - len(s)), 'constant', constant_values=(pad_index, pad_index)) for s in batch]

    return torch.t(torch.LongTensor(batch)), torch.LongTensor(lengths)  # seqlen x bs, bs


def save_model(model: nn.Module, i2s: Dict[int, str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'i2s': i2s
    }
    torch.save(state, path)


def load_model(args: Namespace, model_type=Model) -> Tuple[Model, List[str]]:
    state = torch.load(args.checkpoint_path)
    i2s = state['i2s']
    model = model_type(
        args=args,
        vocab_size=len(i2s),
        pad_index=i2s.index(PAD_TOKEN),
        start_index=i2s.index(START_TOKEN),
        end_index=i2s.index(END_TOKEN)
    )
    model.load_state_dict(state['state_dict'])

    return model, i2s


class LossFunction:
    def __init__(self, pad_index: int = 0, kl_weight: float = 1.0):
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=pad_index).cuda()
        self.kl_weight = kl_weight

    def forward(self,
                output: torch.FloatTensor,
                mu: torch.FloatTensor,
                logvar: torch.FloatTensor,
                target: torch.FloatTensor) -> torch.FloatTensor:
        # dec_seqlen x bs x vocab, bs x vae_latent, bs x vae_latent, dec_seqlen x bs
        reconstruction_loss = self.cross_entropy_loss(output.view(-1, output.size(2)), target.view(-1))

        if mu is not None:
            kld_loss = self.kl_weight * (-0.5) * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / mu.size(0)  # normalize by batch size
        else:
            kld_loss = 0

        return reconstruction_loss + kld_loss

def compute_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


# from https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/convert_parameters.py
def parameters_to_vector(parameters):
    r"""Convert parameters to one vector
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


# from https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/convert_parameters.py
def _check_param_device(param, old_param_device):
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device