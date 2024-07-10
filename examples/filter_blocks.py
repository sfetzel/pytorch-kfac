from torch import Tensor, cat, no_grad
from torch.nn import Linear
import torch
import torch_geometric
from torch_kfac.layers import FullyConnectedFisherBlock, BiasFisherBlock, PyGLinearBlock, FisherBlockFactory, Bias


class TorchLinearBlockFilter(FullyConnectedFisherBlock):
    """
    A linear block for linear layers of a graph neural network.
    The input and the output gradients are filtered by the training mask.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mask = None

    @no_grad()
    def backward_hook(self, module: Linear, grad_inp: Tensor, grad_out: Tensor) -> None:
        assert self.train_mask is not None
        grad_out = (grad_out[0][self.train_mask],)
        super(TorchLinearBlockFilter, self).backward_hook(module, grad_inp, grad_out)

    @no_grad()
    def forward_hook(self, module: Linear, input_data: Tensor, output_data: Tensor) -> None:
        assert self.train_mask is not None
        input_data = (input_data[0][self.train_mask],)
        super(TorchLinearBlockFilter, self).forward_hook(module, input_data, output_data)


class PyGLinearBlockFilter(PyGLinearBlock):
    """
    A linear block for linear layers of a graph neural network.
    The input and the output gradients are filtered by the training mask.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mask = None

    @no_grad()
    def backward_hook(self, module: Linear, grad_inp: Tensor, grad_out: Tensor) -> None:
        assert self.train_mask is not None
        grad_out = (grad_out[0][self.train_mask],)
        super(PyGLinearBlockFilter, self).backward_hook(module, grad_inp, grad_out)

    @no_grad()
    def forward_hook(self, module: Linear, input_data: Tensor, output_data: Tensor) -> None:
        assert self.train_mask is not None
        input_data = (input_data[0][self.train_mask],)
        super(PyGLinearBlockFilter, self).forward_hook(module, input_data, output_data)


class BiasBlockFilter(BiasFisherBlock):
    """
    A linear block for linear layers of a graph neural network.
    The input and the output gradients are filtered by the training mask.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mask = None

    @no_grad()
    def backward_hook(self, module: Linear, grad_inp: Tensor, grad_out: Tensor) -> None:
        assert self.train_mask is not None
        grad_out = (grad_out[0][self.train_mask],)
        super(BiasBlockFilter, self).backward_hook(module, grad_inp, grad_out)

    @no_grad()
    def forward_hook(self, module: Linear, input_data: Tensor, output_data: Tensor) -> None:
        assert self.train_mask is not None
        input_data = (input_data[0][self.train_mask],)
        super(BiasBlockFilter, self).forward_hook(module, input_data, output_data)


FilterBlocksFactory = FisherBlockFactory([
    (torch_geometric.nn.dense.Linear, PyGLinearBlockFilter),
    (torch.nn.Linear, TorchLinearBlockFilter),
    (Bias, BiasBlockFilter)
])