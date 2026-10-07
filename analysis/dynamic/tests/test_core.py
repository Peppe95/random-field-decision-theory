import unittest
import numpy as np
import pandas as pd
from rfdt_correction import core
from rfdt_correction.io import DEFAULT_CONFIG,banks
from rfdt_correction.ddm import joint_bins,p_upper,lower_cdf_sf
from rfdt_correction.simulation import time_bin,seeds_for_cells,simulate_global
from rfdt_correction.validation import check_engine_source

class CoreTests(unittest.TestCase):
    def test_engine_unchanged(self):
        self.assertTrue(check_engine_source()['exact_function_hash_match'])

    def test_exact_tail_endpoints(self):
        c=np.array([.3,1.,2.,3.,4.,10.])
        values=[.2999,.3,.999,1.,4.,10.,10.001]
        expect=[0,1,1,2,5,5,6]
        self.assertEqual([time_bin(x,c) for x in values],expect)

    def test_seed_address_and_prefix(self):
        seen=[]
        for phase in [0,1,16,17]:
            for m in ['RFDT','RFDT_K0']:
                for g in range(3):seen.extend(seeds_for_cells(g,192,63,m,phase).ravel())
        self.assertEqual(len(set(map(int,seen))),len(seen))
        a=seeds_for_cells(7,192,63,'RFDT',0);b=seeds_for_cells(7,256,63,'RFDT',0)
        np.testing.assert_equal(a,b[:192])

    def test_normalization_and_coarsening(self):
        rng=np.random.default_rng(61);cnt=rng.integers(0,20,(8,63,14));base=np.full(cnt.shape,1/14)
        q=core.joint_from_counts(cnt,base,.5);core.check_joint(q)
        pc=q.reshape(8,63,2,7).sum(-1)
        np.testing.assert_allclose(pc.sum(-1),1,atol=1e-14)
        w=core.for_target(q,'window');np.testing.assert_allclose(w.sum(-1),1,atol=1e-14)
        self.assertEqual(float(w[...,0].sum()+w[...,6].sum()+w[...,7].sum()+w[...,13].sum()),0)
        count_coarse=cnt.reshape(8,63,2,7).sum(-1)
        base_coarse=base.reshape(8,63,2,7).sum(-1)
        np.testing.assert_allclose(pc,(count_coarse+.5*base_coarse)/(cnt.sum(-1)[...,None]+.5),atol=1e-14)

    def test_condition_then_mix_is_explicit(self):
        q=np.zeros((2,14));q[0,[0,1,8]]=[.8,.15,.05];q[1,[0,1,8]]=[.2,.1,.7]
        result=core.for_target(q,'window').mean(0)
        alt=core.for_target(q.mean(0,keepdims=True),'window')[0]
        self.assertGreater(float(np.max(abs(result-alt))),.1)
        core.check_joint(result[None])

    def test_gradient(self):
        rng=np.random.default_rng(53);X=rng.random((23,5));L=rng.normal(-15,3,(31,23));L[0]=0
        ob=core.FiniteBankObjective(L,X)
        theta=np.r_[np.full(5,.4),np.full(5,np.log(.21))];grad=ob(theta)[1]
        numeric=[]
        for k in range(10):
            dp=np.zeros(10);dp[k]=1e-6
            numeric.append((ob(theta+dp)[0]-ob(theta-dp)[0])/(2e-6))
        np.testing.assert_allclose(grad,numeric,atol=2e-8,rtol=1e-6)

    def test_optimizer_and_zero_training(self):
        rng=np.random.default_rng(52);X=np.linspace(.01,.99,40)[:,None]
        L=-((X[:,0][None,:]-rng.normal(.3,.04,(25,1)))/.06)**2;L[0]=0
        cfg=DEFAULT_CONFIG['eb'].copy();cfg['maxiter']=300;cfg['starts']=3
        ob=core.FiniteBankObjective(L,X);initial=np.r_[.5,np.log(.28)]
        theta,post,ln,diag=core.fit_finite_bank(L,X,cfg)
        self.assertGreaterEqual(float(ln.sum()),-ob(initial)[0]*len(L)-1e-8)
        self.assertTrue(diag['kkt_pass'])
        np.testing.assert_allclose(post[0],np.exp(core.log_prior(X,theta)),atol=1e-14)

    def test_training_excludes_all_heldout_people(self):
        rng=np.random.default_rng(82);q=rng.dirichlet(np.ones(14),(5,63));E=rng.integers(0,14,(7,63))
        F=np.arange(63)%7
        L=core.training_likelihood(q,E,F,2)
        E2=E.copy();E2[:,F==2]=(E2[:,F==2]+1)%14
        np.testing.assert_equal(L,core.training_likelihood(q,E2,F,2))
        E2[0,:]=-1
        self.assertTrue(np.all(core.training_likelihood(q,E2,F,2)[0]==0))

    def test_ratio_bootstrap_estimand(self):
        sums=np.array([1.,90.,0.]);counts=np.array([1,100,0])
        d,l,h=core.paired_cluster_interval(sums,counts,reps=300,seed=2)
        self.assertAlmostEqual(d,91/101)
        self.assertGreater(abs(d-np.mean([1.,.9])),.04)

    def test_score_from_joint(self):
        rng=np.random.default_rng(5);q=rng.dirichlet(np.ones(14),50);c=rng.integers(0,2,50);b=rng.integers(0,7,50)
        pc,p,lj,lc,lr=core.split_scores(q,c,b)
        np.testing.assert_allclose(lj,lc+lr,atol=1e-14)
        np.testing.assert_allclose(np.exp(lj),p,atol=1e-14)

    def test_ddm_mass_symmetry_and_tolerance(self):
        cut=np.tile([.3,.9,1.5,2.4,4.,10.],(7,1));v=np.tile(np.linspace(-50,50,7),(4,1))
        a=np.array([.1,.5,2.,8.]);z=np.array([.02,.5,.98,.3]);t0=np.array([.01,.15,.29,.2])
        q,m=joint_bins(v,a,z,t0,cut,1e-12);qr,_=joint_bins(-v,a,1-z,t0,cut,1e-14)
        np.testing.assert_allclose(q[:,:,:7],qr[:,:,7:],atol=1e-10)
        np.testing.assert_allclose(q[:,:,7:],qr[:,:,:7],atol=1e-10)
        np.testing.assert_allclose(q.sum(-1),1,atol=1e-14)

    def test_banks_preserve_support_and_prefix(self):
        b=banks(DEFAULT_CONFIG,'base');e=banks(DEFAULT_CONFIG,'expanded')
        self.assertEqual(len(b['RFDT'][0]),288);self.assertEqual(len(b['DDM'][1]),4096)
        for name in b:
            gb,hb,ub=b[name];ge,he,ue=e[name]
            np.testing.assert_allclose(hb.to_numpy(),he.iloc[:len(hb)].to_numpy(),atol=1e-14)
            np.testing.assert_allclose(ub,ue[:len(ub)],atol=1e-14)
        self.assertTrue(np.all(b['RFDT_K0'][0].kappa==0))
        self.assertAlmostEqual(float(b['RFDT_K0'][0].global_prior_mass.sum()),1.)
        self.assertTrue((b['DDM'][1].vcoeff<0).any())

    def test_simulation_reproducible_across_threads(self):
        from numba import set_num_threads
        H=2;P=63;D=np.tile(np.array([-.7,.8]),(H,P,1));pr=np.tile([.5,.5],(P,1))
        cut=np.tile([.3,.9,1.5,2.4,4.,10.],(P,1));h=np.array([[.5,.2,1.,.05],[.8,.3,1.,.1]])
        seeds=seeds_for_cells(0,H,P,'RFDT',0)
        set_num_threads(1);x=simulate_global(.45,.5,80,h,D,pr,cut,10,5760.,seeds)
        set_num_threads(2);y=simulate_global(.45,.5,80,h,D,pr,cut,10,5760.,seeds)
        for a,b in zip(x,y):np.testing.assert_equal(a,b)
        self.assertEqual(int(x[1].sum()),0)
        np.testing.assert_equal(x[0].sum(-1),10)

if __name__=='__main__':unittest.main()
