import unittest

import torch

from torch_kfac.kfac_optimizer import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock
from torch import nn, tensor
from torch.testing import assert_close


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_linear = nn.Linear(5, 2, False)
        self.first_activation = nn.Identity()
        self.second_linear = nn.Linear(2, 2, False)
        self.second_activation = nn.Tanh()
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
        self.model = TestModel()
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
        # y1 = sigma(s1)
        # s2 = s1*W2
        # y2 = sigma(s2)
        # L(y2) = L(sigma(s2)) = L(sigma(s1*W2)) = L(x*W1*W2)
        L = out.sum()
        with self.optimizer.track_backward():
            L.backward()

        # L(y2) = L(sigma(s2)) = sigma(s2_1) + sigma(s2_2).
        # derivative w.r.t to s2:
        # [sigma'(s2_1), sigma'(s2_2)]
        out_first_layer = self.model.first_linear.forward(x)
        out_second_layer = self.model.second_linear.forward(out_first_layer)

        # derivative of tanh(x) is 1-tanh(x)^2.
        expected_sensitivities_2 = 1 - torch.tanh(out_second_layer)**2
        # derivative of L((s2_1, s2_2)) = s2_1 + s2_2 with respect to s2 is [1,1] if activation function is identity.
        # derivative of L((s2_1, s2_2)) = s2_1 + s2_2 with respect to s2 is [L'(s2_1), L'(s2_2)] if
        # activation function is identity.
        assert_close(expected_sensitivities_2, self.optimizer.blocks[3]._sensitivities)

        # derivative of L(y2)=L(sigma(W2 s1))=sigma((W2 s1)_1 + sigma((W2 s1)_2 with respect to s1 is
        # [sigma'((W2 s1)_1*(W2_{00}+W2{10}), sigma'((W2 s1)_2*(W2_{01} + W2_{11})]
        # The first factors in each entry are sigma'(W2 s1).
        sigmas = 1 - torch.tanh(out_second_layer)**2
        # actually the input is multiplied with W^T, therefore here we don't need to transpose.
        expected_s1 = torch.matmul(sigmas, self.model.second_linear.weight)
        assert_close(expected_s1, self.optimizer.blocks[1]._sensitivities)



if __name__ == '__main__':
    unittest.main()
