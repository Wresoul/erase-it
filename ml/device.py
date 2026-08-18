"""Выбор устройства для инференса/обучения: явное имя или автоопределение."""

import torch


def resolve_device(requested: str) -> torch.device:
    """"auto" -> cuda, иначе mps, иначе cpu. Любое другое значение — как есть."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
