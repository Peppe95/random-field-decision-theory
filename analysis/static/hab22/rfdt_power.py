from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import qmc

# Secondary, prospectively frozen robustness model after the completed linear HAB22 analysis.
# The structural regular regime remains kappa < 1. Numerical upper bound is deliberately
# close to 1 rather than the Phase-3b primary .95 screening bound.
POWER_BOUNDS = {
    "alpha": (0.0, 1.0),
    "beta":  (0.05, 4.00),
    "kappa": (0.0, 0.995),
    "lam":   (0.10, 8.00),
    "rho":   (0.05, 1.50),
}


def signed_power(x, rho):
    x = np.asarray(x, float)
    ax = np.abs(x)
    return np.sign(x) * np.power(ax, rho)


def signed_power_drho(x, rho):
    x = np.asarray(x, float)
    ax = np.abs(x)
    out = np.zeros_like(ax, dtype=float)
    nz = ax > 0
    # d[sign(x)|x|^rho]/d rho = sign(x)|x|^rho log|x|.
    out[nz] = np.sign(x[nz]) * np.power(ax[nz], rho) * np.log(ax[nz])
    return out


def build_raw_pair_arrays(A_out, A_prob, B_out, B_prob, max_support=4):
    """Independent/product hypothetical sampling, frozen by the RFDT checkpoint.

    Returns pairwise raw outcome arrays XB-XA and product weights only for dimension checks;
    power utility is applied later to XA/XB separately.
    """
    n = len(A_out)
    XA = np.zeros((n, max_support), float)
    XB = np.zeros((n, max_support), float)
    W = np.zeros((n, max_support), float)
    for j, (ao, ap, bo, bp) in enumerate(zip(A_out, A_prob, B_out, B_prob)):
        vals = []
        for xa, pa in zip(ao, ap):
            for xb, pb in zip(bo, bp):
                vals.append((float(xa), float(xb), float(pa) * float(pb)))
        if len(vals) > max_support:
            raise ValueError(f"Task {j} has {len(vals)} product states > max_support={max_support}")
        for k, (xa, xb, w) in enumerate(vals):
            XA[j, k] = xa; XB[j, k] = xb; W[j, k] = w
        if not np.isclose(W[j].sum(), 1.0, atol=1e-10):
            raise ValueError(f"Task {j} product probabilities sum to {W[j].sum()}")
    return XA, XB, W


def _raw_D_and_drho(XA, XB, rho):
    UA = signed_power(XA, rho)
    UB = signed_power(XB, rho)
    dUA = signed_power_drho(XA, rho)
    dUB = signed_power_drho(XB, rho)
    return UB - UA, dUB - dUA


def power_scale_and_derivative(XA, XB, W, rho):
    """Dataset-wide stimulus-only scale and its piecewise derivative."""
    Draw, dDraw = _raw_D_and_drho(XA, XB, rho)
    rms2 = np.sum(W * Draw * Draw, axis=1)
    rms = np.sqrt(np.maximum(rms2, 0.0))
    n = len(rms)
    if n % 2 != 1:
        raise ValueError("Normalization design must have an odd number of tasks")
    k = n // 2
    mid = int(np.argpartition(rms, k)[k])
    scale = float(rms[mid])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid utility normalization scale {scale}")
    dscale = float(np.sum(W[mid] * Draw[mid] * dDraw[mid]) / rms[mid]) if rms[mid] > 0 else 0.0
    return scale, dscale, mid


def power_D_and_derivative(XA, XB, W, rho, norm_XA=None, norm_XB=None, norm_W=None):
    """Compute comparative evidence and dD/drho under the frozen global scale.

    The normalization design is the entire HAB22 stimulus set, even when fitting only a
    training fold or one participant. Thus held-out *outcomes* never enter preprocessing,
    and the utility scale does not change across CV cells. Held-out task stimuli are used only
    for the prespecified dataset-level, outcome-blind normalization, exactly as in Phase 3b.
    """
    if norm_XA is None:
        norm_XA, norm_XB, norm_W = XA, XB, W
    scale, dscale, mid = power_scale_and_derivative(norm_XA, norm_XB, norm_W, rho)
    Draw, dDraw = _raw_D_and_drho(XA, XB, rho)
    D = Draw / scale
    dD = dDraw / scale - Draw * (dscale / (scale * scale))
    return D, dD, scale, mid

def equilibrium_and_derivatives(D, W, alpha, beta, kappa, dD_drho=None,
                                tol=2e-13, maxiter=100):
    """Unique regular-regime equilibrium and implicit derivatives taskwise.

    Returns dm/d(alpha,beta,kappa[,rho]).
    """
    D = np.asarray(D, float); W = np.asarray(W, float)
    n = D.shape[0]
    lo = np.full(n, -1.0); hi = np.full(n, 1.0); m = np.zeros(n)

    for _ in range(maxiter):
        z = kappa * m[:, None] + beta * D
        t = np.tanh(z)
        Q = np.sum(W * t, axis=1)
        c = np.cosh(z); s = np.sinh(z)
        C = np.sum(W * c, axis=1); S = np.sum(W * s, axis=1); A = S / C
        R = ((1.0-alpha) * np.sum(W*(1.0-t*t), axis=1)
             + alpha * (1.0-A*A))
        G = m - (1.0-alpha)*Q - alpha*A
        Gm = 1.0 - kappa*R
        neg = G < 0
        lo = np.where(neg, m, lo); hi = np.where(~neg, m, hi)
        cand = m - G/Gm
        bad = (cand <= lo) | (cand >= hi) | (~np.isfinite(cand))
        cand = np.where(bad, 0.5*(lo+hi), cand)
        if np.max(np.abs(cand-m)) < tol:
            m = cand; break
        m = cand

    z = kappa*m[:,None] + beta*D
    t = np.tanh(z); sech2 = 1.0-t*t
    Q = np.sum(W*t, axis=1)
    c = np.cosh(z); s = np.sinh(z)
    C = np.sum(W*c, axis=1); S = np.sum(W*s, axis=1); A=S/C
    Esec=np.sum(W*sech2,axis=1)
    R=(1.0-alpha)*Esec + alpha*(1.0-A*A)
    Gm=1.0-kappa*R

    dm_da=(A-Q)/Gm
    Qb=np.sum(W*sech2*D,axis=1)
    EDc=np.sum(W*D*c,axis=1); EDs=np.sum(W*D*s,axis=1)
    Ab=(EDc*C-S*EDs)/(C*C)
    dm_db=((1.0-alpha)*Qb+alpha*Ab)/Gm
    dm_dk=(m*R)/Gm
    derivs=[dm_da,dm_db,dm_dk]

    if dD_drho is not None:
        dzrho = beta * np.asarray(dD_drho, float)
        Qr=np.sum(W*sech2*dzrho,axis=1)
        Sr=np.sum(W*c*dzrho,axis=1)
        Cr=np.sum(W*s*dzrho,axis=1)
        Ar=(Sr*C-S*Cr)/(C*C)
        dm_dr=((1.0-alpha)*Qr+alpha*Ar)/Gm
        derivs.append(dm_dr)
    return m, np.column_stack(derivs)


def unpack_params(x, model):
    x=np.asarray(x,float)
    if model=='power':  return x[0],x[1],x[2],x[3],x[4]
    if model=='rho1':   return x[0],x[1],x[2],x[3],1.0
    if model=='kappa0': return x[0],x[1],0.0,x[2],x[3]
    if model=='alpha0': return 0.0,x[0],x[1],x[2],x[3]
    if model=='alpha1': return 1.0,x[0],x[1],x[2],x[3]
    raise ValueError(model)


def free_bounds(model):
    b=POWER_BOUNDS
    if model=='power': return [b['alpha'],b['beta'],b['kappa'],b['lam'],b['rho']]
    if model=='rho1': return [b['alpha'],b['beta'],b['kappa'],b['lam']]
    if model=='kappa0': return [b['alpha'],b['beta'],b['lam'],b['rho']]
    if model in ('alpha0','alpha1'): return [b['beta'],b['kappa'],b['lam'],b['rho']]
    raise ValueError(model)


def nll_and_grad(x, XA, XB, W, y, n, model='power', norm_XA=None, norm_XB=None, norm_W=None):
    alpha,beta,kappa,lam,rho=unpack_params(x,model)
    D,dD,_,_=power_D_and_derivative(XA,XB,W,rho,norm_XA,norm_XB,norm_W)
    want_rho = model != 'rho1'
    m,dm=equilibrium_and_derivatives(D,W,alpha,beta,kappa,dD if want_rho else None)
    eta=lam*m; p=expit(eta)
    val=np.sum(n*np.logaddexp(0.0,eta)-y*eta)
    resid=n*p-y
    ga=np.sum(resid*lam*dm[:,0])
    gb=np.sum(resid*lam*dm[:,1])
    gk=np.sum(resid*lam*dm[:,2])
    gl=np.sum(resid*m)
    if want_rho:
        gr=np.sum(resid*lam*dm[:,3])
    if model=='power': g=np.array([ga,gb,gk,gl,gr])
    elif model=='rho1': g=np.array([ga,gb,gk,gl])
    elif model=='kappa0': g=np.array([ga,gb,gl,gr])
    elif model in ('alpha0','alpha1'): g=np.array([gb,gk,gl,gr])
    else: raise ValueError(model)
    return float(val),g


def deterministic_starts(model,n_halton=24,seed=20260911):
    """Deterministic multistart set for the final frozen RFDT-P domain.

    Two generic central starts are used when rho is free: one in the low-rho
    region admitted by the training-only HAB22 boundary audit, and one in the
    original broad interior.  This is an optimizer safeguard, not a model
    selection rule; held-out outcomes never determine starts.
    """
    b=np.asarray(free_bounds(model),float); d=len(b)
    if model=='power':
        central=np.array([[0.45,2.50,0.60,2.50,0.22],
                          [0.35,0.90,0.75,2.00,0.85]],float)
    elif model=='rho1':
        central=np.array([[0.45,2.50,0.60,2.50],
                          [0.35,0.90,0.75,2.00]],float)
    elif model=='kappa0':
        central=np.array([[0.45,2.50,2.50,0.22],
                          [0.35,0.90,2.00,0.85]],float)
    else:
        central=np.array([[2.50,0.60,2.50,0.22],
                          [0.90,0.75,2.00,0.85]],float)
    central=np.minimum(np.maximum(central,b[:,0]),b[:,1])
    if n_halton<=0: return central
    sampler=qmc.Halton(d=d,scramble=True,seed=seed)
    pts=qmc.scale(sampler.random(n=n_halton),b[:,0],b[:,1])
    return np.vstack([central,pts])


def fit(XA,XB,W,y,n,model='power',n_halton=24,seed=20260910,
        ftol=1e-11,gtol=2e-7,maxiter=1000,norm_XA=None,norm_XB=None,norm_W=None):
    bounds=free_bounds(model); best=None; recs=[]
    for j,x0 in enumerate(deterministic_starts(model,n_halton,seed)):
        res=minimize(lambda x:nll_and_grad(x,XA,XB,W,y,n,model,norm_XA,norm_XB,norm_W),x0,jac=True,
                     method='L-BFGS-B',bounds=bounds,
                     options={'ftol':ftol,'gtol':gtol,'maxiter':maxiter,'maxls':80})
        recs.append({'start':j,'fun':float(res.fun),'success':bool(res.success),
                     'nit':int(res.nit),'message':str(res.message)})
        if best is None or res.fun < best.fun: best=res
    return best,recs


def predict(XA,XB,W,x,model='power',norm_XA=None,norm_XB=None,norm_W=None):
    a,b,k,lam,rho=unpack_params(x,model)
    D,_,scale,_=power_D_and_derivative(XA,XB,W,rho,norm_XA,norm_XB,norm_W)
    m,_=equilibrium_and_derivatives(D,W,a,b,k)
    return expit(lam*m),m,scale


def gradient_self_test(XA,XB,W):
    # Fixed synthetic counts; no human outcomes involved.
    idx=np.arange(min(31,len(XA)))
    xa,xb,w=XA[idx],XB[idx],W[idx]
    y=np.linspace(5,25,len(idx)); n=np.full(len(idx),30.0)
    x=np.array([0.41,0.88,0.72,2.1,0.83])
    val,g=nll_and_grad(x,xa,xb,w,y,n,'power',XA,XB,W)
    gn=np.zeros_like(g)
    for j in range(len(x)):
        h=1e-6*max(1.0,abs(x[j])); xp=x.copy(); xm=x.copy(); xp[j]+=h; xm[j]-=h
        fp=nll_and_grad(xp,xa,xb,w,y,n,'power',XA,XB,W)[0]
        fm=nll_and_grad(xm,xa,xb,w,y,n,'power',XA,XB,W)[0]
        gn[j]=(fp-fm)/(2*h)
    rel=np.max(np.abs(g-gn)/np.maximum(1.0,np.abs(gn)))
    return {'value':float(val),'max_relative_gradient_error':float(rel),
            'analytic':g.tolist(),'numeric':gn.tolist()}
