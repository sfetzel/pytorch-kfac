import unittest

from torch import Tensor, ones_like
from torch.nn import Linear
from torch.testing import assert_allclose, assert_close

from torch_kfac.layers import FullyConnectedFisherBlock
from torch_kfac.layers.bias_block import BiasFisherBlock
from torch_kfac.layers.bias_layer import Bias


class BiasBlockTest(unittest.TestCase):
    def test_apply_preconditioning(self):
        damping = Tensor([1.])
        #x = Tensor([[1., 2., 3., 4., 5.]])
        x = Tensor([[0., 0., 0., 0., 0.]])
        # weight matrix has size 8 x 5.
        linear_module = Linear(5, 8, bias=True)
        print(linear_module.bias.shape)
        bias_module = Bias(8, 5)
        sensitivities = Tensor([[3., 2., 9., 7., 5., 2., 1., 13.]])
        linear_fisher_block = FullyConnectedFisherBlock(linear_module)
        linear_fisher_block._enable_pi_correction = False
        linear_fisher_block._activations = x
        linear_fisher_block._sensitivities = sensitivities
        linear_fisher_block.update_cov(0.0)
        grads = linear_module.weight, linear_module.bias
        p_weights, p_bias = linear_fisher_block.multiply_preconditioner(grads, damping)

        bias_block = BiasFisherBlock(bias_module)
        bias_block._enable_pi_correction = False
        bias_block._activations = Tensor([[]])
        bias_block._sensitivities = sensitivities
        bias_block.update_cov(0.0)
        (p_bias_block,) = bias_block.multiply_preconditioner([linear_module.bias.reshape(8, 1)], damping)

        assert_close(p_bias_block.reshape(-1), p_bias)

