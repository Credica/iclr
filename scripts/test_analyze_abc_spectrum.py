"""Unit tests for archived ABC spectrum reconstruction; no environment needed."""
import unittest
import numpy as np
import torch
from analyze_abc_spectrum import (NAMES, POLICY, torch_critic, policy_parameters,
                                 entry_state, weight_metrics, legacy_targets)
from analyze_rethink_dynamic import params_from_state, forward, ntk


class ABCSpectrumTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.state = dict(zip(NAMES, [torch.randn(*shape, dtype=torch.float64) * .1 for shape in
            ((5, 7), (5,), (6, 5), (6,), (1, 6), (1,))]))

    def test_torch_numpy_q_match(self):
        x = torch.randn(9, 7, dtype=torch.float64)
        q = torch_critic(self.state, x[:, :3], x[:, 3:])
        actual, _ = forward(params_from_state(self.state), x.numpy())
        np.testing.assert_allclose(q.numpy(), actual, atol=1e-12)

    def test_full_kernel_matches_explicit_jacobian(self):
        state = {k: v.clone().requires_grad_() for k, v in self.state.items()}
        x = torch.randn(8, 7, dtype=torch.float64)
        q = torch_critic(state, x[:, :3], x[:, 3:])
        jac = []
        for value in q:
            grads = torch.autograd.grad(value, tuple(state.values()), retain_graph=True)
            jac.append(torch.cat([g.flatten() for g in grads]))
        jac = torch.stack(jac).numpy()
        p = params_from_state(state); _, cache = forward(p, x.numpy())
        np.testing.assert_allclose(ntk(p, cache), jac @ jac.T / len(x), atol=1e-12)

    def test_entry_does_not_mutate_boundary(self):
        saved = dict(qf1={'x': 1}, qf2={'x': 2}, target_qf1={'x': 3},
                     target_qf2={'x': 4}, policy={'head': 5}, log_alpha=torch.tensor([-3.]))
        entry = entry_state(saved)
        self.assertEqual(entry['target_qf1'], saved['qf1'])
        self.assertIs(entry['policy'], saved['policy'])
        self.assertEqual(entry['log_alpha'].item(), 0.)
        self.assertEqual(saved['log_alpha'].item(), -3.)
        self.assertEqual(saved['target_qf1'], {'x': 3})

    def test_weight_spectrum_scale_and_rank(self):
        w = np.diag([4., 2., 1.])
        s, a = weight_metrics(w); _, b = weight_metrics(10 * w)
        np.testing.assert_array_equal(s, [4., 2., 1.])
        self.assertAlmostEqual(a['stable_rank'], 21 / 16)
        self.assertAlmostEqual(a['entropy_rank'], b['entropy_rank'])
        self.assertAlmostEqual(a['condition'], 4.)

    def test_legacy_policy_heads_and_targets_match_garage(self):
        from garage.torch.modules import GaussianMLPTwoHeadedModule
        from garage.torch.distributions import TanhNormal
        model = GaussianMLPTwoHeadedModule(input_dim=3, output_dim=4, n_tasks=3,
            hidden_sizes=(5, 6), hidden_nonlinearity=torch.nn.ReLU,
            min_std=np.exp(-20), max_std=np.exp(2), normal_distribution_cls=TanhNormal, no_stats=True)
        policy = {'_module.' + k: v for k, v in model.state_dict().items()}
        critic = {k: v.float() for k, v in self.state.items()}
        rng = np.random.RandomState(4)
        bank = dict(observation=rng.randn(7, 3).astype('float32'), next_observation=rng.randn(7, 3).astype('float32'),
            action=rng.randn(7, 4).astype('float32'), reward=rng.randn(7, 1).astype('float32'),
            terminal=np.array([[0], [1], [0], [0], [1], [0], [0]], dtype='float32'))
        obs = torch.from_numpy(bank['next_observation'])
        saved = dict(policy=policy, target_qf1=critic, target_qf2=critic, log_alpha=torch.tensor([-.7]))
        for task in range(3):
            dist = model(obs, seq_idx=task)
            mean, logstd = policy_parameters(policy, obs, task)
            torch.testing.assert_close(mean, dist._normal.base_dist.loc)
            torch.testing.assert_close(logstd.exp(), dist._normal.base_dist.scale)
            actual = legacy_targets(saved, bank, task)
            expected = []
            for index in range(8):
                noise = torch.sin((index + 1) * torch.arange(1, 5).float())[None, :]
                pre = mean + logstd.exp() * noise; action = pre.tanh()
                q = torch_critic(critic, obs, action)
                target = torch.from_numpy(bank['reward']).flatten() + .99 * (1 - torch.from_numpy(bank['terminal']).flatten()) * (
                    q - saved['log_alpha'].exp() * dist.log_prob(action, pre_tanh_value=pre))
                expected.append(target.detach().numpy())
            np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
