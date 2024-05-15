from typing import Optional

from torch import Tensor, empty
from math import sqrt
from torch.nn import Module, Parameter
from torch.nn.init import uniform, zeros_
from torch_geometric.nn.inits import glorot, zeros



class Bias(Module):
    def __init__(self, out_features:int, in_features:Optional[int] = None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.bias = Parameter(empty(out_features, **factory_kwargs))
        self.out_features = out_features
        self.in_features = in_features

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if self.in_features is not None:
            self.bias.uniform_(-1/sqrt(self.in_features), 1/sqrt(self.in_features))
        else:
            zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return input + self.bias
