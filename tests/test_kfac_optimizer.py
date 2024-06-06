import unittest

from torch import nn, float64, tensor
from torch.nn import Linear
from torch.testing import assert_close

from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock, Identity, ConvFisherBlock
from torch_kfac.layers.fisher_block_factory import FisherBlockFactory
from torch_kfac.utils import inverse_by_cholesky


class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(12, 8, 4)
        self.conv2 = nn.Conv2d(9, 7, 2)
        self.conv3 = nn.Conv3d(7, 5, 2)
        self.conv4 = nn.ConvTranspose1d(1, 2, 3)


class KfacOptimizerTest(unittest.TestCase):
    def test_disable_pi_correction(self):
        """
        Checks if the enable_pi_correction parameter is correctly copied to the Fisher blocks.
        """
        model = Linear(3, 4)
        preconditioner = KFAC(model, 0.01, 1e-2, enable_pi_correction=False)
        self.assertEqual(1, len(preconditioner.blocks))
        self.assertFalse(preconditioner.blocks[0]._enable_pi_correction)

    def test_constructor_when_simple_model_should_create_fisher_blocks(self):
        """Tests that the KFAC constructor correctly creates Linear and Convolutional blocks
        for the corresponding layers."""
        model = MockModel()
        optimizer = KFAC(model, 0.01, 1e-2)
        # model.modules(): SimpleModule, Linear, ReLU, Conv1d, Conv2d, Conv3d, ConvTranspose1d.
        self.assertIsInstance(optimizer.blocks[0], Identity)
        self.assertIsInstance(optimizer.blocks[1], FullyConnectedFisherBlock)
        self.assertIsInstance(optimizer.blocks[2], Identity)
        self.assertIsInstance(optimizer.blocks[3], ConvFisherBlock)
        self.assertIsInstance(optimizer.blocks[4], ConvFisherBlock)
        self.assertIsInstance(optimizer.blocks[5], ConvFisherBlock)
        self.assertIsInstance(optimizer.blocks[6], Identity)

    def test_constructor_when_simple_model_with_custom_blocks_should_use_custom_block(self):
        """Tests that the KFAC constructor correctly uses a custom fisher block registered
        in the block factory."""
        model = MockModel()
        factory = FisherBlockFactory()
        factory.register_block(nn.ConvTranspose1d, ConvFisherBlock)
        optimizer = KFAC(model, 0.01, tensor(1e-2), block_factory=factory)
        self.assertIsInstance(optimizer.blocks[6], ConvFisherBlock)

    def setUp(self):
        self.model = Linear(2, 3, dtype=float64)
        self.input = tensor([[1, 0], [0, 1], [0.5, 0.5], [2, 0.5], [4, 1.1]], dtype=float64)
        self.damping = 1e-1
        self.preconditioner = KFAC(self.model, 0, self.damping, update_cov_manually=True)
        self.test_block = self.preconditioner.blocks[0]

    def forward_backward_pass(self):
        with self.preconditioner.track_forward():
            loss = self.model(self.input).norm()
        with self.preconditioner.track_backward():
            loss.backward()

    def calculate_expected_matrices(self):
        """
        Calculates the expected inverse covariance matrices.
        """
        a_damp, s_damp = self.test_block.compute_damping(self.damping, self.test_block.renorm_coeff)
        self.exp_activations_cov_inv = inverse_by_cholesky(self.test_block.activation_covariance, a_damp)
        self.exp_sensitivities_cov_inv = inverse_by_cholesky(self.test_block.sensitivity_covariance, s_damp)

    def test_update_cov_should_update_inverses(self):
        self.forward_backward_pass()
        self.preconditioner.update_cov(True)
        self.calculate_expected_matrices()
        assert_close(self.exp_activations_cov_inv, self.test_block._activations_cov_inv)
        assert_close(self.exp_sensitivities_cov_inv, self.test_block._sensitivities_cov_inv)

    def test_step_should_not_recalculate_inverses(self):
        self.forward_backward_pass()
        self.preconditioner.update_cov(True)
        self.calculate_expected_matrices()
        self.test_block._activations_cov_inv *= 2
        self.test_block._sensitivities_cov_inv *= 1.5
        self.preconditioner.update_cov(False)
        assert_close(self.exp_activations_cov_inv * 2, self.test_block._activations_cov_inv)
        assert_close(self.exp_sensitivities_cov_inv * 1.5, self.test_block._sensitivities_cov_inv)

    def test_step_should_recalculate_inverses(self):
        self.forward_backward_pass()
        self.preconditioner.update_cov(True)
        self.calculate_expected_matrices()
        self.test_block._activations_cov_inv *= 2
        self.test_block._sensitivities_cov_inv *= 1.5
        self.preconditioner.update_cov(True)
        assert_close(self.exp_activations_cov_inv, self.test_block._activations_cov_inv)
        assert_close(self.exp_sensitivities_cov_inv, self.test_block._sensitivities_cov_inv)


if __name__ == '__main__':
    unittest.main()
