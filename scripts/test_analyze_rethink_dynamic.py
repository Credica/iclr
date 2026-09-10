"""Numerical checks independent of the recorded experiments or environments."""
import unittest
import numpy as np
import torch
from analyze_rethink_dynamic import (forward,gradient,jvp,ntk,clipped,Adam,
                                    residual_accounting,spectral_block)
from summarize_rethink_dynamic import (pooled_auc,initial_jump_accounting,success_auc,
                                      matched_block_direction_score)
from analyze_rethink_dynamic_geometry import forced_response


class DynamicAnalysisTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng=np.random.RandomState(123)
        self.p=[self.rng.normal(size=s)*.3 for s in [(7,5),(7,),(6,7),(6,),(1,6),(1,)]]
        self.x=self.rng.normal(size=(8,5));self.y=self.rng.normal(size=8)

    @staticmethod
    def tf(p,x):
        h=torch.relu(x@p[0].T+p[1]);h=torch.relu(h@p[2].T+p[3])
        return (h@p[4].T+p[5]).flatten()

    def test_forward_gradient_kernel_and_jvp(self):
        p=[torch.tensor(a,requires_grad=True) for a in self.p];x=torch.tensor(self.x)
        q=self.tf(p,x);nq,cache=forward(self.p,self.x)
        np.testing.assert_allclose(nq,q.detach().numpy(),rtol=1e-12,atol=1e-12)
        jac=[]
        for i in range(len(x)):
            jac.append(torch.cat([g.flatten() for g in torch.autograd.grad(q[i],p,retain_graph=True)]))
        jac=torch.stack(jac).detach().numpy()
        np.testing.assert_allclose(ntk(self.p,cache),jac@jac.T/len(x),rtol=1e-12,atol=1e-12)
        ng=gradient(self.p,self.x,self.y,cache)
        tg=torch.autograd.grad(((q-torch.tensor(self.y))**2).mean(),p)
        for a,b in zip(ng,tg):np.testing.assert_allclose(a,b.numpy(),rtol=1e-12,atol=1e-12)
        delta=[self.rng.normal(size=a.shape) for a in self.p]
        np.testing.assert_allclose(jvp(self.p,delta,cache),jac@np.concatenate([a.ravel() for a in delta]),rtol=1e-12,atol=1e-12)

    def test_adam_matches_torch_fresh_and_carried(self):
        p=[torch.tensor(a,requires_grad=True) for a in self.p]
        opt=torch.optim.Adam(p,lr=3e-4);npopt=Adam(self.p)
        for step in range(5):
            grads=[self.rng.normal(size=a.shape) for a in self.p]
            before=[a.detach().numpy().copy() for a in p]
            for a,g in zip(p,grads):a.grad=torch.tensor(g)
            opt.step();delta=npopt.update(grads)
            for a,b,c in zip(p,before,delta):np.testing.assert_allclose(a.detach().numpy()-b,c,rtol=1e-10,atol=1e-14)
        carried=Adam(self.p,opt.state_dict())
        grads=[self.rng.normal(size=a.shape) for a in self.p]
        for a,b in zip(npopt.update(grads),carried.update(grads)):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)

    def test_residual_and_energy_identities(self):
        d,u,g,b,lin,change=[self.rng.normal(size=8) for _ in range(6)]
        parts,c,closure,error=residual_accounting(d,u,g,b,lin,change)
        self.assertLess(closure,1e-12);self.assertLess(error,1e-12)
        np.testing.assert_allclose(parts.sum(0),u-change)

    def test_clip_bounds_and_bias_preservation(self):
        original=[p.copy() for p in self.p];c=clipped(self.p)
        for i in (0,2,4):
            s=np.linalg.svd(c[i],compute_uv=False)
            self.assertGreaterEqual(s.min(),.25-1e-10);self.assertLessEqual(s.max(),4+1e-10)
        for i in (1,3,5):np.testing.assert_array_equal(c[i],original[i])
        for a,b in zip(self.p,original):np.testing.assert_array_equal(a,b)

    def test_spectral_block_projection_energy_and_stability(self):
        q=forward(self.p,self.x)[0]
        y=self.rng.normal(size=(6,8))
        s=spectral_block(self.p,self.x,y,q,0,5)
        self.assertAlmostEqual(float(s['drift_energy'].sum()),float(np.square(np.diff(y,axis=0)).sum()),places=10)
        self.assertTrue(np.all(s['retention1000']>=0));self.assertTrue(np.all(s['retention1000']<=1))

    def test_pool_raw_mse_before_panel_ratios(self):
        # Initial panel errors differ by 1000x: averaging panel ratios is wrong.
        branches={'ft_carried':{'initial_mse':[1.,1000.], 'normalized_auc':[1.,1.]},
                  'clip':{'initial_mse':[1.,1000.], 'normalized_auc':[100.,.1]}}
        ratio=pooled_auc([branches],'clip')/pooled_auc([branches],'ft_carried')
        self.assertAlmostEqual(ratio,200./1001.)
        self.assertNotAlmostEqual(ratio,(100.+.1)/2)

    def test_initial_q_jump_energy_identity(self):
        d=self.rng.normal(size=8);j=self.rng.normal(size=8)
        r=initial_jump_accounting(np.mean(d*d),np.mean((d-j)**2),np.mean(j*j))
        self.assertAlmostEqual(r['cross_cost'],-2*np.mean(d*j))
        self.assertAlmostEqual(r['cosine'],np.dot(d,j)/(np.linalg.norm(d)*np.linalg.norm(j)))

    def test_direction_comparison_uses_shared_time_weights(self):
        energy=np.array([[1000.,0.],[0.,1.]])
        slow=np.array([[0.,1.],[0.,1.]])
        weights=np.array([1.,100.])
        result=matched_block_direction_score(energy,slow,weights)
        self.assertAlmostEqual(result,100./101.)
        self.assertAlmostEqual(result,matched_block_direction_score(energy*np.array([[1e8],[.01]]),slow,weights))
        self.assertNotAlmostEqual(result,(energy*slow).sum()/energy.sum())

    def test_success_auc_uses_real_grid_and_deduplicates_exit(self):
        records=[dict(task_env_step=t,train_task_position=1,eval_task_position=1,
                      success_rate=s) for t,s in [(0,0.),(50000,.5),(100000,1.),(100000,1.)]]
        self.assertAlmostEqual(success_auc(records,1,0,100000),.5)
        with self.assertRaises(AssertionError):success_auc(records[:1]+records[2:],1,0,100000)

    def test_forced_response_matches_matrix_recurrence_and_jump_identity(self):
        a=self.rng.normal(size=(8,8));kernel=a@a.T/8
        ev,basis=np.linalg.eigh(kernel);eta=.4/ev[-1]
        d=self.rng.normal(size=8);jump=self.rng.normal(size=8)
        u=self.rng.normal(size=(17,8))
        curve=forced_response(ev,basis,d,u,jump,eta)
        matched=d.copy();common_target=d-jump;propagated=jump.copy()
        operator=np.eye(8)-2*eta*kernel
        for t in range(18):
            expected=[np.mean(matched**2),np.mean(common_target**2),
                      2*np.mean(matched*propagated),np.mean(propagated**2)]
            np.testing.assert_allclose(curve[t],expected,rtol=1e-10,atol=1e-11)
            self.assertAlmostEqual(curve[t,0]-curve[t,1],curve[t,2]-curve[t,3])
            if t<17:
                matched=operator@matched+u[t]
                common_target=operator@common_target+u[t]
                propagated=operator@propagated


if __name__=='__main__':unittest.main()
