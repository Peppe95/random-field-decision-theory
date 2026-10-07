"""Coherent probabilities and the actual finite-bank empirical-Bayes objective.

This is a new implementation of the operations in NEXT_DYNAMIC_ANALYSIS_SPEC.md.
It is NOT the unavailable numerical_repair_core.py from the other chat.
Analytical gradients are checked against independent finite differences in tests.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import qmc

N_BINS = 7
WINDOW_CELLS = np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12], dtype=int)


def check_joint(q, atol=5e-12):
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 14 or not np.all(np.isfinite(q)):
        raise ValueError("Expected a finite 14-cell joint table.")
    if np.any(q < 0):
        raise ValueError("Negative joint probabilities.")
    err = float(np.max(np.abs(q.sum(axis=-1) - 1.0)))
    if err > atol:
        raise ValueError("Joint mass error %.6g exceeds %.6g" % (err, atol))
    pc = q.reshape(q.shape[:-1] + (2, 7)).sum(axis=-1)
    cond = np.divide(q.reshape(q.shape[:-1] + (2, 7)), pc[..., None],
                     out=np.zeros(q.shape[:-1] + (2, 7)), where=pc[..., None] > 0)
    active = pc > 0
    ce = float(np.max(np.abs(cond.sum(axis=-1)[active] - 1))) if np.any(active) else 0.
    if ce > atol:
        raise ValueError("Conditional RT table does not normalize.")
    return {"max_joint_mass_error": err, "max_conditional_mass_error": ce,
            "min_probability": float(q.min()), "max_probability": float(q.max())}


def rfdt_base_measure(nuisance, cutoffs):
    """Uniform over feasible fine cells, zero where RT <= t0+tau_s*bound.

    Since |m| <= 1, an RFDT path cannot reach +/-bound sooner than bound
    internal time units. Numerical pseudo-counts must not create mass below
    that structural minimum. The base depends only on parameters and stimuli.
    """
    h = nuisance
    minimum = (h['t0'].to_numpy() + h['tau_s'].to_numpy()*h['bound'].to_numpy())
    upper = np.column_stack([cutoffs, np.full(len(cutoffs), np.inf)])
    mask = upper[None, :, :] > minimum[:, None, None]
    base = np.concatenate([mask, mask], axis=-1).astype(np.float64)
    base /= base.sum(axis=-1, keepdims=True)
    return base


def joint_from_counts(counts, base, mass=0.5):
    counts = np.asarray(counts)
    if counts.shape != base.shape or counts.shape[-1] != 14:
        raise ValueError("Count/base shape mismatch.")
    if np.any(counts < 0) or np.any(counts[base == 0] != 0):
        raise ValueError("Counts violate nonnegativity or structural RT support.")
    n = counts.sum(axis=-1, keepdims=True)
    if np.any(n <= 0):
        raise ValueError("Empty simulator cell; unresolved paths cannot be discarded.")
    if mass < 0:
        raise ValueError("Negative pseudo-count mass.")
    q = (counts.astype(np.float64) + mass*base)/(n + mass)
    check_joint(q)
    return q


def for_target(q, target):
    """Candidate-level conditioning, before mixing. Keep 14 slots for provenance."""
    if target == 'single_click':
        check_joint(q)
        return q
    if target != 'window':
        raise ValueError("Unknown target %r" % target)
    r = np.zeros_like(q)
    z = q[..., WINDOW_CELLS].sum(axis=-1, keepdims=True)
    if np.any(z <= 0):
        raise ValueError("A candidate has zero probability of window selection.")
    r[..., WINDOW_CELLS] = q[..., WINDOW_CELLS]/z
    check_joint(r)
    return r


def split_scores(q, choices, bins):
    """Every reported component is a log probability from the SAME table."""
    q = np.asarray(q, dtype=float)
    check_joint(q)
    choices = np.asarray(choices, dtype=int)
    bins = np.asarray(bins, dtype=int)
    pc = q.reshape(-1, 2, 7).sum(axis=2)
    k = np.arange(len(q))
    observed = q[k, 7*choices+bins]
    cm = pc[k, choices]
    if np.any(observed <= 0) or np.any(cm <= 0):
        raise FloatingPointError("Zero predictive probability for an observed event; do not floor silently.")
    log_joint = np.log(observed)
    log_choice = np.log(cm)
    log_rt = np.log(observed/cm)
    if not np.allclose(log_joint, log_choice+log_rt, atol=5e-13, rtol=0):
        raise ArithmeticError("Score decomposition is incoherent.")
    return pc, observed, log_joint, log_choice, log_rt


def log_prior(X, theta):
    k = X.shape[1]
    mu, logs = theta[:k], theta[k:]
    a = -0.5*np.sum((X-mu)**2*np.exp(-2*logs), axis=1)
    return a-logsumexp(a)


def projected_gradient(x, grad, bounds):
    g = np.asarray(grad).copy()
    for j, (lo, hi) in enumerate(bounds):
        if (x[j] <= lo+1e-8 and g[j] > 0) or (x[j] >= hi-1e-8 and g[j] < 0):
            g[j] = 0
    return float(np.max(np.abs(g)))


class FiniteBankObjective:
    """Negative log marginal likelihood / canonical participant count.

    loglik[i,h] = sum of TRAINING event log probabilities for participant i.
    w_h = softmax(-0.5 sum_k ((X_hk-mu_k)/s_k)^2).
    L = sum_i log sum_h w_h exp(loglik[i,h]).

    dL = sum_h (sum_i r_ih - n*w_h) * d a_h.
    The -n*w_h term is the finite-bank normalization derivative missing in
    the historical posterior-moment heuristic.
    """
    def __init__(self, loglik, X):
        self.L = np.asarray(loglik, dtype=float)
        self.X = np.asarray(X, dtype=float)
        if self.L.shape[1] != len(self.X):
            raise ValueError("Candidate coordinates and likelihood columns do not match.")
        if np.isnan(self.L).any() or np.isposinf(self.L).any():
            raise ValueError("Invalid training likelihood.")
        self.offset = self.L.max(axis=1)
        if not np.isfinite(self.offset).all():
            bad = np.flatnonzero(~np.isfinite(self.offset))
            raise ValueError("No candidate gives nonzero training likelihood for participants %s" % bad[:20])
        self.A = np.exp(self.L-self.offset[:, None])
        self.n = len(self.L)
        self.k = self.X.shape[1]

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        k = self.k
        dif = self.X-theta[:k]
        invv = np.exp(-2*theta[k:])
        a = -0.5*np.sum(dif*dif*invv, axis=1)
        lp = a-logsumexp(a)
        w = np.exp(lp)
        den = self.A @ w
        if np.min(den) > 1e-200:
            ln = self.offset+np.log(den)
            rh = w*(self.A.T @ (1./den))
        else:
            raw = self.L+lp
            ln = logsumexp(raw, axis=1)
            rh = np.exp(raw-ln[:, None]).sum(axis=0)
        contrast = rh/self.n-w
        grad_mu = -(contrast @ (dif*invv))
        grad_logs = -(contrast @ (dif*dif*invv))
        f = -float(ln.mean())
        g = np.r_[grad_mu, grad_logs]
        if not np.isfinite(f) or not np.isfinite(g).all():
            raise FloatingPointError("Nonfinite finite-bank objective/gradient.")
        return f, g

    def posterior(self, theta):
        raw = self.L+log_prior(self.X, theta)
        normalizer = logsumexp(raw, axis=1)
        post = np.exp(raw-normalizer[:, None])
        post /= post.sum(axis=1, keepdims=True)
        return post, normalizer


def fit_finite_bank(loglik, X, cfg):
    obj = FiniteBankObjective(loglik, X)
    k = X.shape[1]
    bounds = [(0., 1.)]*k + [(np.log(cfg['sd_min']), np.log(cfg['sd_max']))]*k
    starts = [np.r_[np.full(k,.5), np.full(k,np.log(.28))],
              np.r_[np.full(k,.5), np.full(k,np.log(cfg['sd_max']))]]
    # A TRAINING-only posterior moment is a permissible starting point,
    # never treated as an optimizer update or evidence of convergence.
    p0, _ = obj.posterior(starts[1])
    w = p0.mean(axis=0)
    mu = np.clip(w@X, 0, 1)
    sd = np.clip(np.sqrt(np.maximum(w@(X*X)-mu*mu,0)), cfg['sd_min'], cfg['sd_max'])
    starts.append(np.r_[mu, np.log(sd)])
    starts.append(np.r_[np.full(k,.5), np.full(k,np.log(cfg['sd_min']*2))])
    starts = starts[:cfg['starts']]
    records = []
    best = None
    for sn, start in enumerate(starts):
        trace = [float(-obj(start)[0]*obj.n)]
        def callback(x):
            trace.append(float(-obj(x)[0]*obj.n))
        res = minimize(obj, start, jac=True, method='L-BFGS-B', bounds=bounds,
                       callback=callback,
                       options={'maxiter':cfg['maxiter'], 'maxfun':cfg['maxiter']*30,
                                'ftol':cfg['ftol'], 'gtol':cfg['gtol'], 'maxls':50,
                                'maxcor':20})
        f, grad = obj(res.x)
        pg = projected_gradient(res.x, grad, bounds)
        # A second solve of the SAME objective is predeclared if an ftol
        # stop happens before the projected-gradient diagnostic is adequate.
        if pg > cfg['kkt_tolerance']:
            res = minimize(obj, res.x, jac=True, method='L-BFGS-B', bounds=bounds,
                           callback=callback,
                           options={'maxiter':cfg['maxiter']*2, 'maxfun':cfg['maxiter']*60,
                                    'ftol':5e-15, 'gtol':cfg['gtol']/10, 'maxls':70,
                                    'maxcor':25})
            f, grad = obj(res.x)
            pg = projected_gradient(res.x, grad, bounds)
        fallback_used = False
        if pg > cfg['kkt_tolerance']:
            # An independent constrained algorithm, still the exact SAME
            # finite-bank objective, resolves flat/boundary directions where
            # relative function-reduction stopping can be premature.
            fallback_used = True
            res = minimize(obj, res.x, jac=True, method='SLSQP', bounds=bounds,
                           callback=callback,
                           options={'maxiter':max(2000,cfg['maxiter']*3), 'ftol':1e-13})
            f, grad = obj(res.x)
            pg = projected_gradient(res.x, grad, bounds)
        rec = {'start':sn, 'objective_sum':float(-f*obj.n),
               'same_objective_SLSQP_fallback':fallback_used,
               'iterations':int(res.nit), 'evaluations':int(res.nfev),
               'optimizer_success':bool(res.success), 'message':str(res.message),
               'projected_gradient_mean_objective':pg,
               'largest_accepted_objective_drop':float(max(0,-np.min(np.diff(trace)))) if len(trace)>1 else 0.,
               'trace_log_marginal_likelihood':trace, 'theta':res.x.tolist()}
        records.append(rec)
        if best is None or f < best[0]:
            best = f, res.x.copy(), pg
    f, theta, pg = best
    post, ln = obj.posterior(theta)
    prior = np.exp(log_prior(X,theta))
    no_train = np.all(loglik == 0, axis=1)
    if np.any(no_train) and not np.allclose(post[no_train],prior,atol=2e-13,rtol=1e-11):
        raise ArithmeticError("A zero-training participant did not receive the population distribution.")
    diag = {'log_marginal_likelihood':float(ln.sum()),
            'projected_gradient_mean_objective':pg,
            'kkt_pass':bool(pg <= cfg['kkt_tolerance']),
            'mu':theta[:k].tolist(), 'sd':np.exp(theta[k:]).tolist(),
            'mu_at_bound':((theta[:k]<1e-6)|(theta[:k]>1-1e-6)).tolist(),
            'sd_at_bound':((theta[k:]<np.log(cfg['sd_min'])+1e-6)|
                           (theta[k:]>np.log(cfg['sd_max'])-1e-6)).tolist(),
            'n_zero_training_participants':int(no_train.sum()), 'starts':records}
    return theta, post, ln, diag


def training_likelihood(q, events, folds, fold):
    """Sum ONLY training columns. Never use total-minus-test with -inf logs."""
    n = events.shape[0]
    out = np.zeros((n, q.shape[0]), dtype=float)
    with np.errstate(divide='ignore'):
        logq = np.log(q)
    for j in np.flatnonzero(folds != fold):
        idx = np.flatnonzero(events[:,j] >= 0)
        if len(idx):
            out[idx] += logq[:,j,events[idx,j]].T
    return out


def paired_cluster_interval(sums, counts, reps=10000, seed=2026091801):
    """Trial-average estimand: resample participant sums AND participant counts."""
    sums, counts = np.asarray(sums,float), np.asarray(counts,float)
    if len(sums) != len(counts) or counts.sum()<=0:
        raise ValueError("Invalid participant sums/counts.")
    rng = np.random.default_rng(seed)
    values=[]
    for start in range(0, reps, 128):
        ix = rng.integers(0, len(counts), (min(128,reps-start),len(counts)))
        den = counts[ix].sum(axis=1)
        valid=den>0
        values.extend((sums[ix].sum(axis=1)[valid]/den[valid]).tolist())
    lo,hi=np.quantile(values,[.025,.975])
    return float(sums.sum()/counts.sum()), float(lo), float(hi)
