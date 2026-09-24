"""Contrastive loss used by the CCUA objective."""

import torch
from einops import rearrange
from info_nce import info_nce


def uncond_info_nce(query_features, key_features, reduction="none"):
    queries = rearrange(query_features, "b c h w -> b (c h w)")
    keys = rearrange(key_features, "b c h w -> b (c h w)")
    batch_size, dimension = keys.shape
    positive_keys = torch.zeros(batch_size, dimension, device=keys.device)
    negative_keys = torch.zeros(
        batch_size, batch_size - 1, dimension, device=keys.device
    )
    for index in range(batch_size):
        positive_keys[index] = keys[index]
        negative_keys[index] = torch.cat([keys[:index], keys[index + 1:]], dim=0)
    return info_nce(
        queries,
        positive_keys,
        negative_keys,
        reduction=reduction,
        negative_mode="paired",
    )
