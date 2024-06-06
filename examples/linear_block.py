from torch import Tensor, cat, no_grad
from torch.nn import Linear

from torch_kfac.layers import FullyConnectedFisherBlock


class LinearBlock(FullyConnectedFisherBlock):
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
        super(FullyConnectedFisherBlock, self).backward_hook(module, grad_out, grad_inp)

    @no_grad()
    def forward_hook(self, module: Linear, input_data: Tensor, output_data: Tensor) -> None:
        assert self.train_mask is not None
        input_data = (input_data[0][self.train_mask],)
        super(FullyConnectedFisherBlock, self).forward_hook(module, input_data, output_data)
