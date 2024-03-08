import unittest
from torch_kfac.kfac_optimizer import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock
from torch import nn, tensor
from torch.testing import assert_close


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_linear = nn.Linear(5, 2, False)
        self.first_activation = nn.Identity()
        self.second_linear = nn.Linear(2, 2, True)
        self.second_activation = nn.Identity()
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
        # s2 = s1*W2
        # L(s2) = L(s1*W2) = L(x*W1*W2)
        L = out.sum()
        with self.optimizer.track_backward():
            L.backward()

        # derivative of L(s2) with respect to s2 is [1,1].
        assert_close(tensor([[1., 1.]]), self.optimizer.blocks[3]._sensitivities)
        print(self.optimizer.blocks[1]._sensitivities)

        # derivative of L(s1*W2) with respect to s1 is column sum.
        expected_s1 = self.model.second_linear.weight.sum(dim=0).reshape((1,2))
        assert_close(expected_s1, self.optimizer.blocks[1]._sensitivities)



if __name__ == '__main__':
    unittest.main()
