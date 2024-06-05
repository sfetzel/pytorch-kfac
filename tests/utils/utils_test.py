import unittest

import torch
from torch import tensor, mean
from torch.testing import assert_allclose

from torch_kfac.utils import compute_cov


class UtilsTest(unittest.TestCase):

    def test_compute_cov(self):
        # example from https://www.geeksforgeeks.org/covariance-matrix/.
        # rows are samples, columns are features.
        data_matrix = tensor([[80, 70],
                              [63, 20],
                              [100, 50]], dtype=torch.float)
        means = mean(data_matrix, dim=0)
        data_matrix[:, 0] -= means[0]
        data_matrix[:, 1] -= means[1]
        expected_cov = tensor([[343, 260], [260, 633.333]], dtype=torch.float)
        actual_cov = compute_cov(data_matrix, normalizer=2.)
        assert_allclose(actual_cov, expected_cov)
