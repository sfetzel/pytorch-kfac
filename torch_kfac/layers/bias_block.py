from typing import Iterable
import torch
from torch import eye, tensor
from torch.nn import Linear

from .bias_layer import Bias
from .fisher_block import ExtensionFisherBlock, FisherBlock
from ..utils import center, compute_cov, append_homog


class BiasFisherBlock(ExtensionFisherBlock):
    def __init__(self, module: Bias, **kwargs) -> None:
        super().__init__(
            module=module,
            in_features=1,
            out_features=module.out_features,
            dtype=module.bias.dtype,
            device=module.bias.device,
            **kwargs)

        self._activations = None
        self._sensitivities = None
        self._center = False
        #self._activations_cov.value = eye(module.out_features)
        #self._activations_cov.add_to_average(1.0, 0.0)
        #self._activations = tensor([[1.0]])

    @torch.no_grad()
    def forward_hook(self, module: Linear, input_data: torch.Tensor, output_data: torch.Tensor) -> None:
        """
        Captures the input data of this layer, called by torch when the forward function is executed.
        Args:
            module: the linear layer of which to capture the input.
            input_data: the input data of the layer.
            output_data: the output data of the layer.
        """
        x = input_data[0].detach().clone().reshape(-1, self._in_features - self.has_bias).requires_grad_(False)
        if self._activations is None:
            self._activations = x
        else:
            self._activations = torch.cat([self._activations, x])

    @torch.no_grad()
    def backward_hook(self, module: Linear, grad_inp: torch.Tensor, grad_out: torch.Tensor) -> None:
        x = grad_out[0].clone().detach().reshape(-1, self._out_features).requires_grad_(False) * grad_out[0].shape[0]
        if self._sensitivities is None:
            self._sensitivities = x
        else:
            self._sensitivities = torch.cat([self._sensitivities, x])

    def setup(self, center: bool = False, **kwargs):
        super().setup(**kwargs)
        self._center = center

    def update_cov(self, cov_ema_decay: float = 1.0) -> None:
        if self._activations is None or self._sensitivities is None:
            return
        act, sen = self._activations, self._sensitivities
        if self._center:
            act = center(act)
            sen = center(sen)

        if self.has_bias:
            act = append_homog(act)

        activation_cov = compute_cov(act)
        sensitivity_cov = compute_cov(sen)
        self._activations_cov.add_to_average(activation_cov, cov_ema_decay)
        self._sensitivities_cov.add_to_average(sensitivity_cov, cov_ema_decay)
        self._activations = None
        self._sensitivities = None

    def grads_to_mat(self, grads: Iterable[torch.Tensor]) -> torch.Tensor:
        mat_grads = grads[0]
        return mat_grads

    def mat_to_grads(self, mat_grads: torch.Tensor) -> torch.Tensor:
        return mat_grads,

    @property
    def has_bias(self) -> bool:
        return True

    @property
    def vars(self) -> Iterable[torch.Tensor]:
        return [self.module.bias]
