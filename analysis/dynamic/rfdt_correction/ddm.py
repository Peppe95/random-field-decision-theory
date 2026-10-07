"""Tolerance-controlled constant-bound Wiener joint CDF/bin probabilities.

Convention: dX=v dt+dW, boundaries 0,a, start a*z; upper choice = risky.
Short times use the integrated method-of-images density. Long times use the
spectral tail. Bounds on omitted terms control truncation; tighter-tolerance,
reflection, complement and high-precision checks are included in validation.py.
No pseudo-observations or timeout category are added to analytic DDM values.

The CDF construction follows integration of the image/spectral first-passage
representations, not the historical fixed 200-term routine. See NUMERICAL_NOTES.md.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import log_ndtr
from .core import check_joint


def p_upper(v, a, z):
    v,a,z=np.broadcast_arrays(np.asarray(v,float),np.asarray(a,float),np.asarray(z,float))
    t=2*v*a
    out=np.empty_like(t)
    zero=t==0
    out[zero]=z[zero]
    pos=t>0
    out[pos]=np.expm1(-t[pos]*z[pos])/np.expm1(-t[pos])
    neg=t<0
    b=-t[neg]
    out[neg]=np.exp(-b*(1-z[neg]))*np.expm1(-b*z[neg])/np.expm1(-b)
    return out


def _image_cdf(vv, zz, tt, tol):
    """Lower-bound CDF in dimensionless coordinates v*a, z, t/a^2."""
    root=np.sqrt(tt); nu=np.abs(vv)
    def term(y):
        d=np.abs(y)
        la=-vv*zz-nu*d+log_ndtr((nu*tt-d)/root)
        lb=-vv*zz+nu*d+log_ndtr((-nu*tt-d)/root)
        return np.sign(y)*(np.exp(la)+np.exp(lb))
    val=term(zz)
    last_bound=np.full_like(val,np.inf)
    for k in range(1,129):
        val+=term(zz+2*k)+term(zz-2*k)
        # For omitted images |y|>=d, remove exp(-nu^2*s/2)<=1 from
        # the density integral. Each half-line tail is bounded by 2 Phi.
        # Consecutive absolute image bounds have ratio <= exp(-(2d+2)/t).
        d=2*(k+1)-zz
        ratio=np.exp(-(2*d+2)/tt)
        logbound=-vv*zz+np.log(4.)+log_ndtr(-d/root)-np.log1p(-ratio)
        last_bound=np.exp(np.minimum(logbound,700))
        if np.all(last_bound < tol*np.abs(val)+1e-310):
            return val,k,float(last_bound.max())
    raise ArithmeticError("Image-series error bound failed to converge.")


def _image_survivor(vv,zz,tt,tol):
    """Direct image tail for |v*a| >= 1; avoids subtracting CDF from hit mass.

    Integrating one image from t to infinity gives a DIFFERENCE of two
    normal-CDF terms. log/expm1 arithmetic resolves this subtraction. Absolute
    image tails are bounded by their full-time integrals, a geometric series.
    Relative termination matters when conditioning on an extremely rare window.
    """
    root=np.sqrt(tt);nu=np.abs(vv)
    def term(y):
        d=np.abs(y)
        la=-vv*zz-nu*d+log_ndtr((d-nu*tt)/root)
        lb=-vv*zz+nu*d+log_ndtr((-d-nu*tt)/root)
        delta=np.minimum(lb-la,0.)
        val=np.exp(la)*(-np.expm1(delta))
        return np.sign(y)*val
    val=term(zz)
    last_bound=np.full_like(val,np.inf)
    for k in range(1,257):
        val += term(zz+2*k)+term(zz-2*k)
        d=2*(k+1)-zz
        logbound=np.log(2.)-vv*zz-nu*d-np.log(-np.expm1(-2*nu))
        last_bound=np.exp(np.minimum(logbound,700))
        if np.all(last_bound < tol*np.abs(val)+1e-310):
            return val,k,float(last_bound.max())
    raise ArithmeticError("Image survivor relative bound failed to converge.")


def _spectral_tail(vv,zz,tt,tol):
    val=np.zeros_like(vv)
    last_bound=np.full_like(vv,np.inf)
    for n in range(1,513):
        lam=.5*(vv*vv+(n*np.pi)**2)
        val += (np.pi*n/lam)*np.sin(n*np.pi*zz)*np.exp(-vv*zz-lam*tt)
        nn=n+1
        ln=.5*(vv*vv+(nn*np.pi)**2)
        bn=(np.pi*nn/ln)*np.exp(-vv*zz-ln*tt)
        ratio=(1+1/nn)*np.exp(-.5*(2*nn+1)*np.pi**2*tt)
        last_bound=bn/(1-ratio)
        if np.all(last_bound < tol*np.abs(val)+1e-310):
            return val,n,float(last_bound.max())
    raise ArithmeticError("Spectral-series error bound failed to converge.")


def lower_cdf_sf(v,a,z,t,tol=1e-12):
    v,a,z,t=np.broadcast_arrays(np.asarray(v,float),np.asarray(a,float),
                                np.asarray(z,float),np.asarray(t,float))
    if np.any(a<=0) or np.any((z<=0)|(z>=1)):
        raise ValueError("DDM needs a>0 and 0<z<1.")
    hit=p_upper(-v,a,1-z)
    cdf=np.zeros_like(t); sf=hit.copy()
    vv=v*a; tt=t/(a*a)
    positive=t>0
    short=positive & (tt<.12)
    long=positive & ~short
    bounds=[]; image_terms=0; spectral_terms=0
    if short.any():
        val,image_terms,b=_image_cdf(vv[short],z[short],tt[short],tol/8)
        cdf[short]=val; sf[short]=hit[short]-val; bounds.append(b)
        direct=short & (np.abs(vv)>=1.)
        if direct.any():
            survivor,nt,bb=_image_survivor(vv[direct],z[direct],tt[direct],tol/8)
            sf[direct]=survivor;image_terms=max(image_terms,nt);bounds.append(bb)
    if long.any():
        val,spectral_terms,b=_spectral_tail(vv[long],z[long],tt[long],tol/8)
        sf[long]=val; cdf[long]=hit[long]-val; bounds.append(b)
    # Small negative roundoff is measured, never hidden if above tolerance.
    violation=max(float(np.maximum(-cdf,0).max(initial=0)),
                  float(np.maximum(-sf,0).max(initial=0)))
    if violation > max(50*tol,1e-13):
        raise ArithmeticError("DDM CDF/SF positivity violation %.4g" % violation)
    cdf=np.clip(cdf,0,hit); sf=np.clip(sf,0,hit)
    return cdf,sf,hit,{'image_terms_max':image_terms,'spectral_terms_max':spectral_terms,
                      'omitted_series_bound_max':max(bounds,default=0.),
                      'roundoff_correction_max':violation}


def joint_bins(v,a,z,t0,cutoffs,tol=1e-12):
    """v is H x P; cutoffs is P x 6: .3,e1,e2,e3,e4,10.
    Returns H x P x 14, ordered safe bins 0..6 then risky bins 0..6.
    """
    v=np.asarray(v,float)
    a=np.broadcast_to(np.asarray(a,float).reshape(-1,1),v.shape)
    z=np.broadcast_to(np.asarray(z,float).reshape(-1,1),v.shape)
    t0=np.broadcast_to(np.asarray(t0,float).reshape(-1,1),v.shape)
    tt=np.asarray(cutoffs,float)[None,:,:]-t0[:,:,None]
    fields=(np.broadcast_to(v[:,:,None],tt.shape),np.broadcast_to(a[:,:,None],tt.shape),
            np.broadcast_to(z[:,:,None],tt.shape))
    cells=[]; allmeta=[]
    for sign in (0,1):
        vi,ai,zi=fields
        if sign: vi,zi=-vi,1-zi
        F,S,hit,meta=lower_cdf_sf(vi,ai,zi,tt,tol)
        b=np.empty(v.shape+(7,),float)
        b[...,0]=F[...,0]
        for j in range(1,6):
            # Subtract the smaller pair of partial masses, avoiding catastrophic
            # cancellation when a bin is in the early or late tail.
            early=(F[...,j]+F[...,j-1]) <= (S[...,j]+S[...,j-1])
            b[...,j]=np.where(early,F[...,j]-F[...,j-1],S[...,j-1]-S[...,j])
        b[...,6]=S[...,-1]
        neg=float(max(0,-b.min()))
        if neg>max(100*tol,2e-13):
            raise ArithmeticError("Negative DDM interval probability %.5g" % neg)
        b=np.maximum(b,0)
        meta['negative_interval_roundoff_max']=neg
        cells.append(b);allmeta.append(meta)
    q=np.concatenate(cells,axis=-1)
    err=float(np.abs(q.sum(axis=-1)-1).max())
    if err>max(300*tol,1e-12):
        raise ArithmeticError("DDM total mass error %.5g" % err)
    # One primitive table; normalization only absorbs documented floating error.
    q/=q.sum(axis=-1,keepdims=True)
    inv=check_joint(q)
    return q,{'raw_mass_error_max':err,'invariants':inv,'representations':allmeta}


def make_ddm_table(nuisance,design,cutoffs,tol=1e-12,chunk=64):
    from .io import evidence_arrays
    out=np.empty((len(nuisance),len(design),14),float)
    checks=[]
    for start in range(0,len(nuisance),chunk):
        h=nuisance.iloc[start:start+chunk]
        ev=[]
        for rho in h.rho:
            D,pr,scale=evidence_arrays(design,float(rho));ev.append((D*pr).sum(axis=1))
        ev=np.asarray(ev)
        v=h.vmax.to_numpy()[:,None]*np.tanh(.5*h.vcoeff.to_numpy()[:,None]*ev)
        q,meta=joint_bins(v,h.a.to_numpy(),h.z.to_numpy(),h.t0.to_numpy(),cutoffs,tol)
        out[start:start+len(h)]=q
        checks.append(meta)
    return out,checks
