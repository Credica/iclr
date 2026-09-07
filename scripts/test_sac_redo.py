"""Unit checks for periodic SAC ReDo neuron recycling."""
import unittest

import torch

from garage.torch.algos.sac import SAC
from garage.torch.modules import MLPModule


class ReDoChecks(unittest.TestCase):

    def test_normalized_dormant_units_are_recycled_and_disconnected(self):
        network = MLPModule(
            input_dim=2, output_dim=1, hidden_sizes=(3, 3),
            ReDo=True, no_stats=False)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # Populate Adam state so the test also covers selective moment reset.
        loss = network(torch.ones(4, 2))[0].sum()
        loss.backward()
        optimizer.step()

        network._features = [
            torch.tensor([[0., 1., 1.], [0., 1., 1.]]),
            torch.tensor([[1., 0., 1.], [1., 0., 1.]]),
        ]
        algo = SAC.__new__(SAC)
        algo._redo_tau = 0.1

        recycled = algo._redo_network(network, optimizer)

        self.assertEqual(2, recycled)
        self.assertTrue(torch.equal(
            network._layers[1][0].weight[:, 0], torch.zeros(3)))
        self.assertTrue(torch.equal(
            network._output_layers[0][0].weight[:, 1], torch.zeros(1)))
        first_weight_state = optimizer.state[
            network._layers[0][0].weight]['exp_avg']
        self.assertTrue(torch.equal(first_weight_state[0], torch.zeros(2)))

    def test_empty_feature_state_is_a_noop(self):
        network = MLPModule(
            input_dim=2, output_dim=1, hidden_sizes=(3, 3),
            ReDo=True, no_stats=False)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        algo = SAC.__new__(SAC)
        algo._redo_tau = 0.1
        self.assertEqual(0, algo._redo_network(network, optimizer))


if __name__ == '__main__':
    unittest.main()
