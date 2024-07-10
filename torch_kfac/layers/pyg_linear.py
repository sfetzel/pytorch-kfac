from typing import Iterable
import torch
from torch_geometric.nn.dense import Linear

from . import FullyConnectedFisherBlock
from .fisher_block import ExtensionFisherBlock, FisherBlock
from ..utils import center, compute_cov, append_homog


class PyGLinearBlock(FullyConnectedFisherBlock):
    def __init__(self, module: Linear, **kwargs) -> None:
        super().__init__(
            module=module,
            in_features=module.in_channels,
            out_features=module.out_channels,
            **kwargs)

