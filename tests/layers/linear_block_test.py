import unittest

import torch

from torch_kfac.kfac_optimizer import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock
from torch import nn, tensor
from torch.testing import assert_close


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_linear = nn.Linear(5, 2, True)
        self.first_activation = nn.Sigmoid()
        self.first_activation_d = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))  # derivative of sigmoid
        self.second_linear = nn.Linear(2, 2, True)
        self.second_activation = nn.Tanh()
        self.second_activation_d = lambda x: 1 - torch.tanh(x) ** 2  # derivative of Tanh
        self.layers = nn.Sequential(
            self.first_linear,
            self.first_activation,
            self.second_linear,
            self.second_activation,
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class FullyConnectedFisherBlockTest(unittest.TestCase):

    def setUp(self):
        self.model = MockModel()
        self.optimizer = KFAC(self.model, 9e-3, 1e-3)

    def test_constructor(self):
        self.assertIsInstance(self.optimizer.blocks[1], FullyConnectedFisherBlock)
        self.assertIsInstance(self.optimizer.blocks[3], FullyConnectedFisherBlock)

    def test_forward_hook_two_layers(self):
        """
        Tests if the forward hook correctly captures the input data (activations
        from previous layer).
        """
        in_data = tensor([[1., 2., 3., 4., 5.],
                          [6., 7., 8., 9., 10.]])
        in_data_second = self.model.first_activation.forward(self.model.first_linear.forward(in_data))
        with self.optimizer.track_forward():
            self.model(in_data)

        assert_close(in_data, self.optimizer.blocks[1]._activations)
        assert_close(in_data_second, self.optimizer.blocks[3]._activations)

    def test_backward_hook(self):
        """
        Tests if the backward hook correctly captures the gradient with
        respect to the input data.
        """
        x = tensor([[1., 2., 3., 4., 5.]])
        out = self.model.forward(x)
        # W1 is weight matrix of layer 1, W2 is weight matrix of layer 2.
        # L(y1, y2) = y1 + y2
        # s1 = x*W1
        # a1 = sigma1(s1)
        # s2 = a1*W2
        # a2 = sigma2(s2)
        # L(a2) = L(sigma2(s2)) = L(sigma2(a1*W2^T)) = L(sigma2(sigma1(x * W1^T)*W2^T))
        L = out.sum()
        with self.optimizer.track_backward():
            L.backward()

        # L(a2) = L(sigma2(s2)) = sigma2(s2_1) + sigma2(s2_2).
        out_first_linear = self.model.first_linear.forward(x)
        out_first_layer = self.model.first_activation.forward(out_first_linear)
        out_second_linear = self.model.second_linear.forward(out_first_layer)

        # derivative of L(a2) w.r.t to s2:
        # [sigma2'(s2_1), sigma2'(s2_2)]
        expected_sensitivities_2 = self.model.second_activation_d(out_second_linear)
        assert_close(expected_sensitivities_2, self.optimizer.blocks[3]._sensitivities)

        # derivative of L(a2) = L(sigma1(a1*W2^T)) = sigma((tanh(s1) * W2^T)_1 + sigma((tanh(s1) * W2^T)_2 with respect to s1.
        sigmas = self.model.first_activation_d(out_first_linear)

        # actually the input is multiplied with W^T, therefore here we don't need to transpose.
        expected_s1 = self.model.first_activation_d(out_first_linear) * torch.matmul(
            self.model.second_activation_d(out_second_linear), self.model.second_linear.weight)
        assert_close(expected_s1, self.optimizer.blocks[1]._sensitivities)

    def test_activations_covariance_should_add_homg_coordinate(self):
        x = tensor([[1., 2., 3., 4., 5.],
                    [2., 3., 4., 5., 6.]])
        with self.optimizer.track_forward():
            self.model.forward(x)
        self.optimizer.blocks[1]._sensitivities = tensor([[1, 2]])
        self.optimizer.blocks[1].update_cov(0.0)

        act_with_hom = tensor([[1., 2., 3., 4., 5., 1.],
                               [2., 3., 4., 5., 6., 1.]]).T
        expected_act_cov = (act_with_hom @ act_with_hom.T) / 2
        assert_close(self.optimizer.blocks[1].activation_covariance, expected_act_cov)


if __name__ == '__main__':
    unittest.main()
