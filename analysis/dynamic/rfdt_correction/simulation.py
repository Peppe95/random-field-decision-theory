"""Chunked, restartable event-count simulation using the unchanged RFDT engine."""
from __future__ import annotations
import numpy as np
from numba import njit,prange
from .engine import _simulate_one_trial


@njit(cache=True)
def time_bin(rt,cut):
    if rt<cut[0]:return 0
    for k in range(1,5):
        if rt<cut[k]:return k
    if rt<=cut[5]:return 5
    return 6


@njit(cache=True,parallel=True)
def simulate_global(alpha,kappa,N,harr,Dbank,probs,cutoffs,M,horizon,seeds):
    H,P=Dbank.shape[0],Dbank.shape[1]
    counts=np.zeros((H,P,14),np.uint32)
    unresolved=np.zeros((H,P),np.uint32)
    beyond90=np.zeros((H,P),np.uint32)
    max_tau=np.zeros((H,P),np.float64)
    for hp in prange(H*P):
        h=hp//P;j=hp-h*P
        beta,bound,tau_s,t0=harr[h]
        np.random.seed(seeds[h,j])
        for m in range(M):
            c,tau=_simulate_one_trial(Dbank[h,j],probs[j],2,alpha,beta,kappa,
                                      N,bound,1.0,horizon)
            if c==0:
                unresolved[h,j]+=1
                continue
            if tau>90:beyond90[h,j]+=1
            if tau>max_tau[h,j]:max_tau[h,j]=tau
            rt=t0+tau_s*tau
            b=time_bin(rt,cutoffs[j])
            ci=0 if c<0 else 1
            counts[h,j,7*ci+b]+=1
    return counts,unresolved,beyond90,max_tau


def seeds_for_cells(g, h_count, p_count, model, phase):
    """Distinct 32-bit seeds via bijection of distinct reserved cell IDs.

    512 globals x 512 nuisance x 63 problems per RFDT family; 128 independent
    phase namespaces. Enlarging a bank retains existing cell streams exactly.
    Different simulator families/phases do not reuse cell seeds.
    """
    if g>=512 or h_count>512 or p_count!=63 or not 0<=phase<128:
        raise ValueError('Seed address exceeds reserved domain.')
    family=0 if model=='RFDT' else 1
    index=(np.uint64(phase)*(1<<25)+np.uint64(family)*(1<<24)+
           np.uint64(g)*512*63+np.arange(h_count,dtype=np.uint64)[:,None]*63+
           np.arange(p_count,dtype=np.uint64)[None,:])
    if np.max(index)>=2**32:raise ValueError('Seed address overflow.')
    return ((index*np.uint64(1664525)+np.uint64(1013904223))%np.uint64(2**32)).astype(np.uint32)
