"""Data-independent checks for empirical subspace and prospective lag analysis."""
import unittest
import numpy as np
from analyze_rethink_subspace import (covariance, demand_family, eigen,
    shape_response, demand_score, tracking_summary)


class SubspaceTest(unittest.TestCase):
    def test_opposite_demands_do_not_cancel(self):
        d = np.array([[2., 0.], [-2., 0.]])
        np.testing.assert_array_equal(covariance(d), np.diag([4., 0.]))
        _, subspace, desc = demand_family(d)
        self.assertEqual(desc['rank95'], 1)
        self.assertEqual(desc['mean_vector_energy_fraction'], 0.)
        self.assertAlmostEqual(float(np.square(d @ subspace).sum()), 8.)

    def test_covariance_rotation_and_energy(self):
        rng = np.random.RandomState(9)
        d = rng.normal(size=(17, 5))
        q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        np.testing.assert_allclose(covariance(d @ q), q.T @ covariance(d) @ q, atol=1e-12)
        self.assertAlmostEqual(float(np.trace(covariance(d))), float(np.square(d).sum() / 17))

    def test_orthogonal_mixing_of_demand_columns_preserves_burden(self):
        rng = np.random.RandomState(17)
        d = rng.normal(size=(9, 5))
        mixing, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        np.testing.assert_allclose(covariance(mixing @ d), covariance(d), atol=1e-12)
        c, s, _ = demand_family(d); c2, s2, _ = demand_family(mixing @ d)
        g = dict(eigenvalues=np.array([.001, .1, .5, 1., 5.]), eigenvectors=np.eye(5))
        a, b = demand_score(c, s, g, .01), demand_score(c2, s2, g, .01)
        self.assertAlmostEqual(a['ridge_0.001']['shape_burden'], b['ridge_0.001']['shape_burden'])

    def test_eigensystem_rejects_materially_indefinite_kernel(self):
        with self.assertRaises(AssertionError): eigen(np.diag([-.001, 1.]))

    def test_inverse_burden_matches_direct_solve(self):
        rng = np.random.RandomState(4)
        a = rng.normal(size=(5, 5)); k = a @ a.T + .1 * np.eye(5)
        d = rng.normal(size=(9, 5)); c, space, _ = demand_family(d)
        val, vec = eigen(k); geom = dict(eigenvalues=val, eigenvectors=vec)
        score = demand_score(c, space, geom, .1 / val[-1])
        expected = np.trace(np.linalg.solve(k / val.mean() + .001 * np.eye(5), c)) / np.trace(c)
        self.assertAlmostEqual(score['ridge_0.001']['shape_burden'], expected)

    def test_shape_is_scale_invariant_raw_is_not(self):
        d = np.array([[1., 2.], [2., -1.]])
        c, space, _ = demand_family(d)
        a = demand_score(c, space, dict(eigenvalues=np.array([.1, 1.]), eigenvectors=np.eye(2)), .001)
        b = demand_score(c, space, dict(eigenvalues=np.array([1., 10.]), eigenvectors=np.eye(2)), .001)
        self.assertAlmostEqual(a['shape_remaining1000'], b['shape_remaining1000'])
        self.assertAlmostEqual(a['ridge_0.001']['shape_burden'], b['ridge_0.001']['shape_burden'])
        self.assertNotAlmostEqual(a['raw_remaining1000'], b['raw_remaining1000'])

    def test_tiny_slow_energy_can_dominate_burden(self):
        d = np.diag(np.sqrt([.01, .99]))
        c, space, _ = demand_family(d)
        score = demand_score(c, space, dict(eigenvalues=np.array([1e-9, 2.]), eigenvectors=np.eye(2)), .01)
        self.assertLess(score['slow_energy_fraction'], .02)
        self.assertGreater(score['ridge_0.001']['slow_burden_fraction'], .9)

    def test_stability_and_fixed_dimension_reference(self):
        _, eta, ret, slow = shape_response(np.array([1e-9, .1, .2, 30.]))
        self.assertEqual(eta, .45 / 4)
        self.assertTrue(np.all((ret >= 0) & (ret <= 1)))
        self.assertTrue(slow[0]); self.assertFalse(slow[-1])

    def test_projected_tracking_exact_energy_identity(self):
        rng = np.random.RandomState(16)
        parts = rng.normal(size=(5, 12, 4))
        d = np.r_[rng.normal(size=(1, 4)), np.zeros((12, 4))]
        d[1:] = d[0] + np.cumsum(parts.sum(axis=0), axis=0)
        basis, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        result, coeff, proj = tracking_summary(d, parts, basis, np.array([1, 1, 0, 0], dtype=bool), left=3)
        for group in result['groups'].values(): self.assertLess(group['signed_energy_identity_abs'], 1e-10)
        g = result['groups']
        self.assertAlmostEqual(g['slow']['final_mse'] + g['fast']['final_mse'], g['all']['final_mse'])
        np.testing.assert_allclose(np.diff(coeff, axis=0), proj.sum(axis=0), atol=1e-12)

    def test_prefix_subspace_does_not_use_suffix(self):
        prefix = np.array([[1., 0., 0.], [2., 0., 0.]])
        _, space, _ = demand_family(prefix)
        suffix = np.array([[0., 10., 0.]])
        self.assertAlmostEqual(float(np.square(suffix @ space).sum()), 0.)

    def test_actual_slow_mode_retains_more_error_in_known_linear_system(self):
        kernel = np.array([.01, 1.])
        d = np.empty((31, 2)); d[0] = 1.
        parts = np.zeros((5, 30, 2))
        for t in range(30):
            parts[1, t] = -.1 * kernel * d[t]
            d[t + 1] = d[t] + parts[1, t]
        result, _, _ = tracking_summary(d, parts, np.eye(2), np.array([True, False]), left=3)
        slow, fast = result['groups']['slow'], result['groups']['fast']
        self.assertGreater(slow['final_over_initial'], fast['final_over_initial'])
        self.assertAlmostEqual(slow['correction_gain'], .001)
        self.assertAlmostEqual(fast['correction_gain'], .1)
        self.assertAlmostEqual(slow['signed_delta_mse']['target'], 0.)


if __name__ == '__main__': unittest.main()
