#!/usr/bin/env python3
"""
RFDT v2 — Phase 2b.1
Direct synthetic parameter recovery for the exact finite-N dynamic first-passage model.

Purpose
-------
1) Generate synthetic choice + RT data from the exact Gillespie RFDT dynamics.
2) Fit the SAME dynamic model in two ways:
      A. choice only: marginal first-passage choice probabilities;
      B. choice + RT: joint first-passage choice × RT-bin probabilities.
3) Compare recovery of (alpha, beta, kappa), with special attention to kappa.
4) Separate the regular beta regime from the known weak-beta singular regime.

This is deliberately a controlled fixed-nuisance recovery study:
    N=120, bound=0.20, r=1, tau_s=1, t0=0.20
are fixed at their generating values. Nuisance-parameter confounds belong in Phase 2b.2.

The likelihood is simulator-based but NOT a summary-information/GMM objective:
it is a multinomial likelihood for the discretized first-passage distribution.
Choice-only is the exact marginal of the same dynamic simulator (up to MC estimation).

Input
-----
Pass the Phase-1b robust-design CSV used by Phase 2a:
    python run_rfdt_phase2b_recovery_v3.py --design robust_design_problems.csv

The script imports rfdt_phase2a.load_phase1b_design so the stimulus
representation is exactly the same as in the validated Phase-2a analysis.

The exact Phase-2a loader requires the five validated problem IDs:
    rnd_0277, rnd_0145, rnd_0101, rnd_0199, rnd_0159

It tries several common column names for the D support and its probabilities.
If it cannot infer the schema, it stops and prints the detected columns rather
than silently using the wrong representation.

Outputs
-------
rfdt_phase2b_recovery_output/
    config.json
    problems_used.csv
    rt_bin_edges.csv
    truth_parameters.csv
    truth_behavior.csv
    recovery_estimates.csv
    recovery_summary.csv
    recovery_by_truth.csv
    atlas_diagnostics.csv
    nearest_atlas_floor.csv
    recovery_alpha.png
    recovery_beta.png
    recovery_kappa.png
    kappa_absolute_error.png
    README_RESULTS.txt

and a ZIP with the same contents.

Notes
-----
- Fixed N_A = round(alpha*N), as required by the Phase-2a correction.
- Initial spins are unbiased.
- Initial fields are sampled from the objective D distribution for both groups.
- Persistent fields remain fixed.
- Adaptive fields refresh with rho_k(sigma) ∝ p_k exp(beta*sigma*d_k).
- Spin heat-bath uses m_{-i}, not m.
- Z integrates m exactly between CTMC jumps and the decision boundary can be
  crossed inside a waiting interval.
"""

import argparse
import ast
import io
import json
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import qmc
from numba import njit, prange, set_num_threads, get_num_threads

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SELECT_IDS = ["rnd_0277", "rnd_0145", "rnd_0101", "rnd_0199", "rnd_0159"]

# Exact dynamic nuisance settings inherited from Phase 2a.
N_UNITS = 120
BOUND = 0.20
R_ADAPT = 1.0
TAU_S = 1.0
T0 = 0.20
MAX_INTERNAL_TIME = 90.0

# Search domain. beta intentionally includes the analytically singular/sloppy region.
ALPHA_RANGE = (0.08, 0.92)
BETA_RANGE = (0.20, 1.60)
KAPPA_RANGE = (0.08, 0.90)

# Pilot-quality defaults. Scale these later for the final recovery package.
N_ATLAS = 1024
N_SIM_ATLAS = 1000
N_RT_BINS = 5

# RT-bin construction uses an independent prior-predictive pilot.
N_BIN_THETA = 48
N_BIN_SIM_PER_PROBLEM = 120

# Synthetic recovery design.
N_REGULAR_TRUTHS = 8       # includes nominal truth plus 7 QMC points
N_WEAK_TRUTHS = 4
N_REPLICATES = 2
N_OBS_PER_PROBLEM = 120

# Dirichlet/Jeffreys smoothing of simulator-estimated probabilities.
PSEUDO_CHOICE = 0.5
PSEUDO_JOINT = 0.5

# Seeds are intentionally separated by task.
SEED_ATLAS = 52001
SEED_BINS = 52002
SEED_TRUTHS = 52003
SEED_DATA = 52004

# Nominal Phase-2a point.
NOMINAL = (0.45, 0.90, 0.55)


# ---------------------------------------------------------------------
# Utility / parameter transforms
# ---------------------------------------------------------------------

def _logit(x):
    return np.log(x / (1.0 - x))

def _inv_logit(x):
    return expit(x)

def physical_to_eta(theta):
    a, b, k = theta
    return np.array([_logit(a), np.log(b), _logit(k)], dtype=float)

def eta_to_physical(eta):
    return np.array([_inv_logit(eta[0]), np.exp(eta[1]), _inv_logit(eta[2])], dtype=float)

ETA_LO = physical_to_eta((ALPHA_RANGE[0], BETA_RANGE[0], KAPPA_RANGE[0]))
ETA_HI = physical_to_eta((ALPHA_RANGE[1], BETA_RANGE[1], KAPPA_RANGE[1]))

def unit_to_theta(u, eta_lo=ETA_LO, eta_hi=ETA_HI):
    eta = eta_lo + np.asarray(u) * (eta_hi - eta_lo)
    return eta_to_physical(eta)

def theta_to_unit(theta, eta_lo=ETA_LO, eta_hi=ETA_HI):
    eta = physical_to_eta(theta)
    return (eta - eta_lo) / (eta_hi - eta_lo)

def sobol_points(n, seed, eta_lo=ETA_LO, eta_hi=ETA_HI):
    # Generate a power-of-two Sobol block and slice it, preserving balance properties
    # without scipy warnings for arbitrary requested n.
    if n <= 0:
        return np.empty((0, 3), dtype=float)
    engine = qmc.Sobol(d=3, scramble=True, seed=seed)
    m = int(math.ceil(math.log2(n)))
    u = engine.random_base2(m)[:n]
    return np.vstack([unit_to_theta(row, eta_lo, eta_hi) for row in u])


# ---------------------------------------------------------------------
# Exact Phase-2a problem loader / provenance audit
# ---------------------------------------------------------------------

def load_problems(path):
    """
    Load the Phase-1b robust design through the SAME loader used by Phase 2a.

    This intentionally imports rfdt_phase2a.py rather than reparsing the
    design independently.  It also verifies the stored nominal static
    diagnostics when those columns are present.
    """
    path = Path(path)

    try:
        from rfdt_phase2a import (
            load_phase1b_design,
            StructuralParams as RefStructuralParams,
            static_diagnostics as ref_static_diagnostics,
        )
    except Exception as e:
        raise RuntimeError(
            "Could not import rfdt_phase2a.py. Put this Phase-2b script in "
            "the same directory as rfdt_phase2a.py."
        ) from e

    problems, phase1df = load_phase1b_design(path)

    by_label = {pr.label: pr for pr in problems}
    missing = [x for x in SELECT_IDS if x not in by_label]
    if missing:
        raise RuntimeError(
            "The supplied robust-design file is not the Phase-1b bank used "
            f"by the validated Phase 2a run. Missing labels: {missing}"
        )

    # Strict provenance check against nominal values saved by Phase 1b.
    ref_theta = RefStructuralParams(alpha=.45, beta=.90, kappa=.55)
    audit_rows = []
    lower_cols = {c.lower(): c for c in phase1df.columns}

    for pid in SELECT_IDS:
        pr = by_label[pid]
        sd = ref_static_diagnostics(pr, ref_theta)
        row = phase1df.loc[phase1df["label"].astype(str) == pid].iloc[0]

        rec = {
            "label": pid,
            "Q0_recomputed": float(sd["Q0"]),
            "V_recomputed": float(sd["V"]),
            "QV_recomputed": float(sd["QV"]),
        }

        for key, colname in [
            ("Q0", "nominal_q0"),
            ("V", "nominal_v"),
            ("QV", "nominal_qv"),
        ]:
            if colname in lower_cols:
                stored = float(row[lower_cols[colname]])
                recomputed = float(sd[key])
                rec[f"{key}_stored"] = stored
                rec[f"{key}_absdiff"] = abs(stored - recomputed)
                if not np.isclose(stored, recomputed, rtol=2e-8, atol=2e-10):
                    raise RuntimeError(
                        f"Provenance audit failed for {pid} {key}: "
                        f"stored={stored:.12g}, recomputed={recomputed:.12g}. "
                        "Do not continue: this may not be the exact Phase-1b design."
                    )
        audit_rows.append(rec)

    # Preserve exactly the validated Phase-2a selected order.
    rows = []
    for pid in SELECT_IDS:
        pr = by_label[pid]
        rows.append((
            pid,
            np.asarray(pr.d, dtype=float).copy(),
            np.asarray(pr.p, dtype=float).copy(),
        ))

    return rows, phase1df, pd.DataFrame(audit_rows), str(path)


def pack_problems(rows):
    P = len(rows)
    Kmax = max(len(d) for _, d, _ in rows)
    D = np.zeros((P, Kmax), dtype=np.float64)
    probs = np.zeros((P, Kmax), dtype=np.float64)
    K = np.zeros(P, dtype=np.int64)
    ids = []
    for j, (pid, d, p) in enumerate(rows):
        ids.append(pid)
        K[j] = len(d)
        D[j, :len(d)] = d
        probs[j, :len(p)] = p
    return ids, D, probs, K


# ---------------------------------------------------------------------
# Exact finite-N Gillespie first-passage simulator
# ---------------------------------------------------------------------

@njit(cache=True)
def _stable_logistic2(z):
    # 1 / (1 + exp(-2z)), stable for large |z|.
    x = 2.0 * z
    if x >= 0.0:
        e = math.exp(-x) if x < 745.0 else 0.0
        return 1.0 / (1.0 + e)
    else:
        e = math.exp(x) if x > -745.0 else 0.0
        return e / (1.0 + e)


@njit(cache=True)
def _sample_multinomial_seq(n, probs, K, out):
    remaining_n = n
    remaining_p = 1.0
    for k in range(K - 1):
        if remaining_n <= 0:
            out[k] = 0
            continue
        pk = probs[k] / remaining_p if remaining_p > 0.0 else 0.0
        if pk < 0.0:
            pk = 0.0
        elif pk > 1.0:
            pk = 1.0
        x = np.random.binomial(remaining_n, pk)
        out[k] = x
        remaining_n -= x
        remaining_p -= probs[k]
    if K > 0:
        out[K - 1] = remaining_n


@njit(cache=True)
def _rho_for_spin(d, probs, K, beta, sigma, out):
    # Stable softmax of log p_k + beta*sigma*d_k.
    mx = -1e300
    for k in range(K):
        if probs[k] > 0.0:
            v = math.log(probs[k]) + beta * sigma * d[k]
            if v > mx:
                mx = v
    s = 0.0
    for k in range(K):
        if probs[k] <= 0.0:
            out[k] = 0.0
        else:
            v = math.exp(math.log(probs[k]) + beta * sigma * d[k] - mx)
            out[k] = v
            s += v
    if s <= 0.0:
        for k in range(K):
            out[k] = probs[k]
    else:
        for k in range(K):
            out[k] /= s


@njit(cache=True)
def _simulate_one_trial(d, probs, K, alpha, beta, kappa,
                        N, bound, r_adapt, max_time):
    """
    Returns:
        choice: -1 lower, +1 upper, 0 timeout
        tau: internal first-passage time (or max_time if timeout)
    """
    NA = int(np.rint(alpha * N))
    if NA < 0:
        NA = 0
    elif NA > N:
        NA = N
    NP = N - NA

    # Counts by field k and spin state.
    pc = np.zeros(K, dtype=np.int64)
    ac = np.zeros(K, dtype=np.int64)
    Pm = np.zeros(K, dtype=np.int64)
    Pp = np.zeros(K, dtype=np.int64)
    Am = np.zeros(K, dtype=np.int64)
    Ap = np.zeros(K, dtype=np.int64)

    # Initial field distribution is p_k for both persistent and adaptive groups.
    _sample_multinomial_seq(NP, probs, K, pc)
    _sample_multinomial_seq(NA, probs, K, ac)

    # Unbiased initial spins.
    S = 0  # sum_i sigma_i
    for k in range(K):
        nplus = np.random.binomial(pc[k], 0.5)
        Pp[k] = nplus
        Pm[k] = pc[k] - nplus
        S += 2 * nplus - pc[k]

        nplus = np.random.binomial(ac[k], 0.5)
        Ap[k] = nplus
        Am[k] = ac[k] - nplus
        S += 2 * nplus - ac[k]

    rho_m = np.zeros(K, dtype=np.float64)
    rho_p = np.zeros(K, dtype=np.float64)
    _rho_for_spin(d, probs, K, beta, -1.0, rho_m)
    _rho_for_spin(d, probs, K, beta, +1.0, rho_p)

    t = 0.0
    Z = 0.0

    while t < max_time:
        m = S / float(N)

        # Because m is constant between CTMC jumps, boundary crossing can be exact.
        hit_dt = 1e300
        hit_choice = 0
        if m > 0.0:
            h = (bound - Z) / m
            if h >= 0.0:
                hit_dt = h
                hit_choice = 1
        elif m < 0.0:
            h = (-bound - Z) / m
            if h >= 0.0:
                hit_dt = h
                hit_choice = -1

        total = 0.0

        # Spin-flip rates. Heat-bath uses m_{-i}.
        for k in range(K):
            # Current sigma=-1 -> +1.
            m_minus_i = (S + 1.0) / float(N)
            p_plus = _stable_logistic2(kappa * m_minus_i + beta * d[k])
            total += (Pm[k] + Am[k]) * p_plus

            # Current sigma=+1 -> -1.
            m_minus_i = (S - 1.0) / float(N)
            p_plus = _stable_logistic2(kappa * m_minus_i + beta * d[k])
            total += (Pp[k] + Ap[k]) * (1.0 - p_plus)

        # Adaptive field-refresh state changes.
        if r_adapt > 0.0 and NA > 0:
            for k in range(K):
                if Am[k] > 0:
                    total += Am[k] * r_adapt * (1.0 - rho_m[k])
                if Ap[k] > 0:
                    total += Ap[k] * r_adapt * (1.0 - rho_p[k])

        if total <= 0.0:
            if hit_dt < 1e299 and t + hit_dt <= max_time:
                return hit_choice, t + hit_dt
            return 0, max_time

        dt = -math.log(max(np.random.random(), 1e-300)) / total

        # First passage occurs before the next state-changing event.
        if hit_dt <= dt and t + hit_dt <= max_time:
            return hit_choice, t + hit_dt

        if t + dt > max_time:
            Z += m * (max_time - t)
            return 0, max_time

        Z += m * dt
        t += dt

        # Select one state-changing event.
        u = np.random.random() * total
        cum = 0.0
        done = False

        for k in range(K):
            # Persistent: - -> +
            m_minus_i = (S + 1.0) / float(N)
            p_plus = _stable_logistic2(kappa * m_minus_i + beta * d[k])
            cum += Pm[k] * p_plus
            if u < cum:
                Pm[k] -= 1
                Pp[k] += 1
                S += 2
                done = True
                break

            # Persistent: + -> -
            m_minus_i = (S - 1.0) / float(N)
            p_plus = _stable_logistic2(kappa * m_minus_i + beta * d[k])
            cum += Pp[k] * (1.0 - p_plus)
            if u < cum:
                Pp[k] -= 1
                Pm[k] += 1
                S -= 2
                done = True
                break

            # Adaptive spin: - -> +
            m_minus_i = (S + 1.0) / float(N)
            p_plus = _stable_logistic2(kappa * m_minus_i + beta * d[k])
            cum += Am[k] * p_plus
            if u < cum:
                Am[k] -= 1
                Ap[k] += 1
                S += 2
                done = True
                break

            # Adaptive spin: + -> -
            m_minus_i = (S - 1.0) / float(N)
            p_plus = _stable_logistic2(kappa * m_minus_i + beta * d[k])
            cum += Ap[k] * (1.0 - p_plus)
            if u < cum:
                Ap[k] -= 1
                Am[k] += 1
                S -= 2
                done = True
                break

        if done:
            continue

        # Adaptive field refresh. Aggregate rate excludes self-resampling;
        # conditional destination is sampled from rho excluding current k.
        if r_adapt > 0.0 and NA > 0:
            for k in range(K):
                if Am[k] > 0:
                    rate = Am[k] * r_adapt * (1.0 - rho_m[k])
                    cum += rate
                    if u < cum:
                        mass = 1.0 - rho_m[k]
                        uu = np.random.random() * mass
                        cc = 0.0
                        dest = -1
                        for kp in range(K):
                            if kp == k:
                                continue
                            cc += rho_m[kp]
                            if uu < cc:
                                dest = kp
                                break
                        if dest < 0:
                            for kp in range(K - 1, -1, -1):
                                if kp != k and rho_m[kp] > 0:
                                    dest = kp
                                    break
                        if dest >= 0:
                            Am[k] -= 1
                            Am[dest] += 1
                        done = True
                        break

                if Ap[k] > 0:
                    rate = Ap[k] * r_adapt * (1.0 - rho_p[k])
                    cum += rate
                    if u < cum:
                        mass = 1.0 - rho_p[k]
                        uu = np.random.random() * mass
                        cc = 0.0
                        dest = -1
                        for kp in range(K):
                            if kp == k:
                                continue
                            cc += rho_p[kp]
                            if uu < cc:
                                dest = kp
                                break
                        if dest < 0:
                            for kp in range(K - 1, -1, -1):
                                if kp != k and rho_p[kp] > 0:
                                    dest = kp
                                    break
                        if dest >= 0:
                            Ap[k] -= 1
                            Ap[dest] += 1
                        done = True
                        break

        # Floating-point cumulative-rounding fallback: simply continue.
        # It should be vanishingly rare and does not change state.
    return 0, max_time


@njit(cache=True)
def _simulate_raw_seeded(d, probs, K, alpha, beta, kappa, n_trials,
                         N, bound, r_adapt, tau_s, t0, max_time, seed):
    np.random.seed(seed)
    choices = np.empty(n_trials, dtype=np.int8)
    rts = np.empty(n_trials, dtype=np.float64)
    for i in range(n_trials):
        c, tau = _simulate_one_trial(
            d, probs, K, alpha, beta, kappa, N, bound, r_adapt, max_time
        )
        choices[i] = c
        rts[i] = t0 + tau_s * tau
    return choices, rts


@njit(cache=True, parallel=True)
def _build_atlas_counts(theta_bank, D, probs, Karr, rt_edges,
                        n_sim, N, bound, r_adapt, tau_s, t0, max_time, seed_base):
    C = theta_bank.shape[0]
    P = D.shape[0]
    B = rt_edges.shape[1] - 1
    joint = np.zeros((C, P, 2 * B + 1), dtype=np.int64)
    choice = np.zeros((C, P, 3), dtype=np.int64)  # lower, upper, timeout

    for ci in prange(C):
        alpha = theta_bank[ci, 0]
        beta = theta_bank[ci, 1]
        kappa = theta_bank[ci, 2]

        for pj in range(P):
            # Re-seed per candidate/problem so results are stable if thread count changes.
            np.random.seed(seed_base + ci * 100003 + pj * 1009)
            K = Karr[pj]
            d = D[pj, :K]
            p = probs[pj, :K]

            for _ in range(n_sim):
                c, tau = _simulate_one_trial(
                    d, p, K, alpha, beta, kappa, N, bound, r_adapt, max_time
                )
                if c == 0:
                    joint[ci, pj, 2 * B] += 1
                    choice[ci, pj, 2] += 1
                else:
                    rt = t0 + tau_s * tau
                    # Fixed problem-specific bins.
                    b = B - 1
                    for bb in range(B):
                        if rt < rt_edges[pj, bb + 1]:
                            b = bb
                            break
                    if c < 0:
                        joint[ci, pj, b] += 1
                        choice[ci, pj, 0] += 1
                    else:
                        joint[ci, pj, B + b] += 1
                        choice[ci, pj, 1] += 1
    return joint, choice


# ---------------------------------------------------------------------
# RT bins, truth set, synthetic datasets
# ---------------------------------------------------------------------

def build_rt_bins(D, probs, Karr):
    P = D.shape[0]
    # Independent QMC points used only to define fixed bins.
    theta_bin = sobol_points(N_BIN_THETA, SEED_BINS)
    all_rt = [[] for _ in range(P)]

    for ti, th in enumerate(theta_bin):
        for pj in range(P):
            seed = SEED_BINS + ti * 10007 + pj * 101
            c, rt = _simulate_raw_seeded(
                D[pj, :Karr[pj]], probs[pj, :Karr[pj]], Karr[pj],
                th[0], th[1], th[2],
                N_BIN_SIM_PER_PROBLEM,
                N_UNITS, BOUND, R_ADAPT, TAU_S, T0, MAX_INTERNAL_TIME,
                seed
            )
            rt = rt[c != 0]
            if len(rt):
                all_rt[pj].append(rt)

    edges = np.empty((P, N_RT_BINS + 1), dtype=float)
    for pj in range(P):
        if not all_rt[pj]:
            raise RuntimeError(f"No non-timeout RTs while constructing bins for problem {pj}.")
        x = np.concatenate(all_rt[pj])
        # Interior prior-predictive quantiles. Infinite outer edges prevent truncation.
        q = np.quantile(x, np.linspace(0, 1, N_RT_BINS + 1)[1:-1])
        # Enforce strictly increasing edges if rounded/degenerate.
        for k in range(1, len(q)):
            if q[k] <= q[k-1]:
                q[k] = np.nextafter(q[k-1], np.inf)
        edges[pj, 0] = -np.inf
        edges[pj, -1] = np.inf
        edges[pj, 1:-1] = q
    return edges


def _restricted_eta_bounds(alpha_rng, beta_rng, kappa_rng):
    lo = physical_to_eta((alpha_rng[0], beta_rng[0], kappa_rng[0]))
    hi = physical_to_eta((alpha_rng[1], beta_rng[1], kappa_rng[1]))
    return lo, hi


def build_truths():
    # Regular regime: deliberately away from beta<.5 singularity and hard boundaries.
    reg_lo, reg_hi = _restricted_eta_bounds(
        (0.18, 0.82), (0.55, 1.35), (0.16, 0.84)
    )
    reg_qmc = sobol_points(max(N_REGULAR_TRUTHS - 1, 0), SEED_TRUTHS, reg_lo, reg_hi)

    truths = [{
        "truth_id": "regular_nominal",
        "regime": "regular",
        "alpha": NOMINAL[0],
        "beta": NOMINAL[1],
        "kappa": NOMINAL[2],
    }]
    for i, th in enumerate(reg_qmc):
        truths.append({
            "truth_id": f"regular_{i+1:02d}",
            "regime": "regular",
            "alpha": float(th[0]),
            "beta": float(th[1]),
            "kappa": float(th[2]),
        })

    # Explicit weak-beta stress cases. We expect structural degradation here.
    weak = [
        (0.30, 0.25, 0.35),
        (0.30, 0.35, 0.75),
        (0.70, 0.25, 0.75),
        (0.70, 0.35, 0.35),
    ][:N_WEAK_TRUTHS]

    for i, th in enumerate(weak):
        truths.append({
            "truth_id": f"weakbeta_{i+1:02d}",
            "regime": "weak_beta",
            "alpha": th[0],
            "beta": th[1],
            "kappa": th[2],
        })
    return pd.DataFrame(truths)


def rt_bin_index(rt, edges):
    # edges: [-inf, ..., +inf]
    return int(np.searchsorted(edges[1:-1], rt, side="right"))


def simulate_observed_dataset(theta, D, probs, Karr, rt_edges, replicate):
    P = D.shape[0]
    B = N_RT_BINS
    joint_counts = np.zeros((P, 2 * B + 1), dtype=np.int64)
    choice_counts = np.zeros((P, 3), dtype=np.int64)
    behavior_rows = []

    for pj in range(P):
        seed = SEED_DATA + replicate * 1000003 + pj * 1009 + int(
            1000 * theta[0] + 10000 * theta[1] + 100000 * theta[2]
        )
        c, rt = _simulate_raw_seeded(
            D[pj, :Karr[pj]], probs[pj, :Karr[pj]], Karr[pj],
            theta[0], theta[1], theta[2],
            N_OBS_PER_PROBLEM,
            N_UNITS, BOUND, R_ADAPT, TAU_S, T0, MAX_INTERNAL_TIME,
            seed
        )

        for cc, rr in zip(c, rt):
            if cc == 0:
                joint_counts[pj, 2 * B] += 1
                choice_counts[pj, 2] += 1
            else:
                b = rt_bin_index(rr, rt_edges[pj])
                if cc < 0:
                    joint_counts[pj, b] += 1
                    choice_counts[pj, 0] += 1
                else:
                    joint_counts[pj, B + b] += 1
                    choice_counts[pj, 1] += 1

        good = c != 0
        behavior_rows.append({
            "problem_index": pj,
            "p_upper": float(np.mean(c == 1)),
            "p_lower": float(np.mean(c == -1)),
            "p_timeout": float(np.mean(c == 0)),
            "mean_rt": float(np.mean(rt[good])) if np.any(good) else np.nan,
            "median_rt": float(np.median(rt[good])) if np.any(good) else np.nan,
        })

    return joint_counts, choice_counts, behavior_rows


# ---------------------------------------------------------------------
# Likelihood atlas and recovery
# ---------------------------------------------------------------------

def smoothed_log_probs(counts, pseudo):
    K = counts.shape[-1]
    denom = counts.sum(axis=-1, keepdims=True) + pseudo * K
    probs = (counts + pseudo) / denom
    return np.log(probs), probs


def score_atlas(obs_counts, atlas_log_probs):
    # obs_counts: P x Kcells; atlas: Ccandidate x P x Kcells
    return np.einsum("pk,cpk->c", obs_counts, atlas_log_probs, optimize=True)


def nearest_atlas(theta, theta_bank):
    u = theta_to_unit(theta)
    ubank = np.vstack([theta_to_unit(x) for x in theta_bank])
    dist = np.sqrt(np.sum((ubank - u[None, :]) ** 2, axis=1))
    idx = int(np.argmin(dist))
    return idx, float(dist[idx])


def summarize_recovery(rec):
    rows = []
    for regime in ["regular", "weak_beta", "all"]:
        sub0 = rec if regime == "all" else rec[rec["regime"] == regime]
        for method in ["choice", "choice_rt"]:
            sub = sub0[sub0["method"] == method]
            if len(sub) == 0:
                continue
            for par in ["alpha", "beta", "kappa"]:
                true = sub[f"true_{par}"].to_numpy(float)
                est = sub[f"est_{par}"].to_numpy(float)
                err = est - true
                corr = np.corrcoef(true, est)[0, 1] if np.std(true) > 0 and np.std(est) > 0 else np.nan
                rows.append({
                    "regime": regime,
                    "method": method,
                    "parameter": par,
                    "n": len(sub),
                    "bias": float(np.mean(err)),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err ** 2))),
                    "corr_true_est": float(corr) if np.isfinite(corr) else np.nan,
                })

    out = pd.DataFrame(rows)

    # Add direct choice-vs-RT gain rows for quick inspection.
    gains = []
    for regime in ["regular", "weak_beta", "all"]:
        for par in ["alpha", "beta", "kappa"]:
            a = out[(out.regime == regime) & (out.parameter == par)]
            if len(a) != 2:
                continue
            r_choice = float(a.loc[a.method == "choice", "rmse"].iloc[0])
            r_joint = float(a.loc[a.method == "choice_rt", "rmse"].iloc[0])
            gains.append({
                "regime": regime,
                "parameter": par,
                "rmse_choice": r_choice,
                "rmse_choice_rt": r_joint,
                "rmse_ratio_choice_over_choice_rt": r_choice / r_joint if r_joint > 0 else np.inf,
            })
    return out, pd.DataFrame(gains)


def recovery_by_truth(rec):
    rows = []
    for tid, g in rec.groupby("truth_id"):
        base = {
            "truth_id": tid,
            "regime": g["regime"].iloc[0],
            "true_alpha": g["true_alpha"].iloc[0],
            "true_beta": g["true_beta"].iloc[0],
            "true_kappa": g["true_kappa"].iloc[0],
        }
        for par in ["alpha", "beta", "kappa"]:
            for method in ["choice", "choice_rt"]:
                s = g[g.method == method]
                err = s[f"est_{par}"].to_numpy() - s[f"true_{par}"].to_numpy()
                base[f"{par}_{method}_mae"] = float(np.mean(np.abs(err)))
                base[f"{par}_{method}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_recovery(rec, outdir, par):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for method, marker in [("choice", "o"), ("choice_rt", "x")]:
        s = rec[(rec["regime"] == "regular") & (rec["method"] == method)]
        ax.scatter(s[f"true_{par}"], s[f"est_{par}"], label=method, marker=marker, alpha=0.75)
    vals = np.concatenate([
        rec[f"true_{par}"].to_numpy(float),
        rec[f"est_{par}"].to_numpy(float)
    ])
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = 0.05 * max(hi - lo, 1e-8)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")
    ax.set_xlabel(f"True {par}")
    ax.set_ylabel(f"Recovered {par}")
    ax.set_title(f"RFDT Phase 2b recovery: {par} (regular beta)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"recovery_{par}.png", dpi=180)
    plt.close(fig)


def plot_kappa_abs_error(rec, outdir):
    keys = ["truth_id", "replicate"]
    c = rec[rec.method == "choice"].set_index(keys)
    j = rec[rec.method == "choice_rt"].set_index(keys)
    idx = c.index.intersection(j.index)
    x = np.abs(c.loc[idx, "est_kappa"].to_numpy() - c.loc[idx, "true_kappa"].to_numpy())
    y = np.abs(j.loc[idx, "est_kappa"].to_numpy() - j.loc[idx, "true_kappa"].to_numpy())

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(x, y, alpha=0.8)
    mx = max(np.max(x), np.max(y), 1e-6)
    ax.plot([0, mx], [0, mx], linestyle="--")
    ax.set_xlabel("|kappa error|: choice only")
    ax.set_ylabel("|kappa error|: choice + RT")
    ax.set_title("Does RT improve kappa recovery?")
    fig.tight_layout()
    fig.savefig(outdir / "kappa_absolute_error.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design", "--problems", dest="design", type=str, default=None,
        help=(
            "Phase-1b robust-design CSV used by Phase 2a. "
            "Both --design and the legacy --problems spelling are accepted."
        )
    )
    parser.add_argument(
        "--outdir", type=str, default="rfdt_phase2b_recovery_output"
    )
    parser.add_argument(
        "--threads", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--atlas", type=int, default=N_ATLAS,
        help="Number of QMC parameter points in the simulator likelihood atlas."
    )
    parser.add_argument(
        "--sim-atlas", type=int, default=N_SIM_ATLAS,
        help="Simulator trials per atlas point per problem."
    )
    args = parser.parse_args()

    set_num_threads(max(1, int(args.threads)))
    print(f"Numba threads: {get_num_threads()}")

    if args.design is None:
        candidates = [
            Path("phase1b_robust_design_problems.csv"),
            Path("robust_design_problems.csv"),
        ]
        found = [p for p in candidates if p.exists()]
        if not found:
            raise RuntimeError(
                "No --design path supplied and the Phase-1b robust-design CSV "
                "was not found in the current directory.\n"
                "Example:\n"
                "  python run_rfdt_phase2b_recovery_v3.py "
                "--design robust_design_problems.csv"
            )
        problem_path = found[0]
    else:
        problem_path = Path(args.design)

    outdir = Path(args.outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n=== 1. Load exact Phase-2a validated stimuli ===")
    rows, phase1df, provenance_audit, source_name = load_problems(problem_path)
    ids, D, probs, Karr = pack_problems(rows)
    print(f"Source: {source_name}")
    print(f"Problems: {ids}")
    print("Provenance audit: PASS — nominal Q0/V/QV reproduce the Phase-1b file.")
    provenance_audit.to_csv(outdir / "phase1b_provenance_audit.csv", index=False)

    prob_rows = []
    for pid, d, p in rows:
        prob_rows.append({
            "problem_id": pid,
            "d_support": json.dumps([float(x) for x in d]),
            "probabilities": json.dumps([float(x) for x in p]),
            "K": len(d),
        })
    pd.DataFrame(prob_rows).to_csv(outdir / "problems_used.csv", index=False)

    print("\n=== 2. Compile simulator / construct fixed RT bins ===")
    # Force JIT compilation before timing.
    _ = _simulate_raw_seeded(
        D[0, :Karr[0]], probs[0, :Karr[0]], Karr[0],
        NOMINAL[0], NOMINAL[1], NOMINAL[2],
        2, N_UNITS, BOUND, R_ADAPT, TAU_S, T0, MAX_INTERNAL_TIME, 123
    )

    t = time.time()
    rt_edges = build_rt_bins(D, probs, Karr)
    print(f"RT-bin pilot finished in {time.time()-t:.1f}s")

    rtrows = []
    for pj, pid in enumerate(ids):
        for b in range(N_RT_BINS):
            rtrows.append({
                "problem_id": pid,
                "bin": b,
                "lower": rt_edges[pj, b],
                "upper": rt_edges[pj, b+1],
            })
    pd.DataFrame(rtrows).to_csv(outdir / "rt_bin_edges.csv", index=False)

    print("\n=== 3. Build simulator likelihood atlas ===")
    theta_bank = sobol_points(int(args.atlas), SEED_ATLAS)
    t = time.time()
    joint_counts_atlas, choice_counts_atlas = _build_atlas_counts(
        theta_bank, D, probs, Karr, rt_edges,
        int(args.sim_atlas),
        N_UNITS, BOUND, R_ADAPT, TAU_S, T0, MAX_INTERNAL_TIME,
        SEED_ATLAS
    )
    atlas_sec = time.time() - t
    print(f"Atlas: {len(theta_bank)} theta points x {len(ids)} problems x "
          f"{int(args.sim_atlas)} trials = "
          f"{len(theta_bank)*len(ids)*int(args.sim_atlas):,} trials")
    print(f"Atlas simulation finished in {atlas_sec:.1f}s")

    logp_joint, p_joint = smoothed_log_probs(joint_counts_atlas, PSEUDO_JOINT)
    logp_choice, p_choice = smoothed_log_probs(choice_counts_atlas, PSEUDO_CHOICE)

    timeout = choice_counts_atlas[:, :, 2] / float(args.sim_atlas)
    pd.DataFrame([{
        "n_atlas": len(theta_bank),
        "n_sim_per_problem": int(args.sim_atlas),
        "mean_timeout_rate": float(timeout.mean()),
        "median_timeout_rate": float(np.median(timeout)),
        "p95_timeout_rate": float(np.quantile(timeout, .95)),
        "max_timeout_rate": float(timeout.max()),
        "fraction_candidate_problem_timeout_gt_1pct": float(np.mean(timeout > .01)),
        "atlas_seconds": float(atlas_sec),
    }]).to_csv(outdir / "atlas_diagnostics.csv", index=False)

    print("\n=== 4. Build truth set and synthetic datasets ===")
    truths = build_truths()
    truths.to_csv(outdir / "truth_parameters.csv", index=False)

    floor_rows = []
    for _, tr in truths.iterrows():
        th = np.array([tr.alpha, tr.beta, tr.kappa], dtype=float)
        idx, du = nearest_atlas(th, theta_bank)
        est = theta_bank[idx]
        floor_rows.append({
            "truth_id": tr.truth_id,
            "regime": tr.regime,
            "nearest_atlas_index": idx,
            "unit_distance": du,
            "true_alpha": th[0], "nearest_alpha": est[0], "abs_alpha": abs(est[0]-th[0]),
            "true_beta": th[1], "nearest_beta": est[1], "abs_beta": abs(est[1]-th[1]),
            "true_kappa": th[2], "nearest_kappa": est[2], "abs_kappa": abs(est[2]-th[2]),
        })
    pd.DataFrame(floor_rows).to_csv(outdir / "nearest_atlas_floor.csv", index=False)

    recovery_rows = []
    behavior_rows_all = []

    for ti, tr in truths.iterrows():
        theta = np.array([tr.alpha, tr.beta, tr.kappa], dtype=float)
        for rep in range(N_REPLICATES):
            joint_obs, choice_obs, beh = simulate_observed_dataset(
                theta, D, probs, Karr, rt_edges,
                replicate=rep + 1000 * ti
            )

            for row in beh:
                row.update({
                    "truth_id": tr.truth_id,
                    "regime": tr.regime,
                    "replicate": rep,
                    "problem_id": ids[row["problem_index"]],
                    "true_alpha": theta[0],
                    "true_beta": theta[1],
                    "true_kappa": theta[2],
                })
                behavior_rows_all.append(row)

            ll_choice = score_atlas(choice_obs, logp_choice)
            ll_joint = score_atlas(joint_obs, logp_joint)

            for method, ll in [("choice", ll_choice), ("choice_rt", ll_joint)]:
                idx = int(np.argmax(ll))
                est = theta_bank[idx]
                recovery_rows.append({
                    "truth_id": tr.truth_id,
                    "regime": tr.regime,
                    "replicate": rep,
                    "method": method,
                    "atlas_index": idx,
                    "loglik": float(ll[idx]),
                    "loglik_gap_to_second": float(
                        ll[idx] - np.partition(ll, -2)[-2]
                    ),
                    "true_alpha": theta[0],
                    "true_beta": theta[1],
                    "true_kappa": theta[2],
                    "est_alpha": est[0],
                    "est_beta": est[1],
                    "est_kappa": est[2],
                    "err_alpha": est[0] - theta[0],
                    "err_beta": est[1] - theta[1],
                    "err_kappa": est[2] - theta[2],
                })

        print(f"  completed {tr.truth_id}: "
              f"alpha={theta[0]:.3f}, beta={theta[1]:.3f}, kappa={theta[2]:.3f}")

    rec = pd.DataFrame(recovery_rows)
    beh = pd.DataFrame(behavior_rows_all)
    rec.to_csv(outdir / "recovery_estimates.csv", index=False)
    beh.to_csv(outdir / "truth_behavior.csv", index=False)

    summary, gains = summarize_recovery(rec)
    # Put the direct gain rows below the ordinary summary in a single CSV-friendly table.
    summary.to_csv(outdir / "recovery_summary.csv", index=False)
    gains.to_csv(outdir / "recovery_rmse_gains.csv", index=False)
    recovery_by_truth(rec).to_csv(outdir / "recovery_by_truth.csv", index=False)

    print("\n=== 5. Core recovery comparison ===")
    print(summary.to_string(index=False))
    print("\nRMSE ratios (choice / choice+RT): values >1 favor adding RT")
    print(gains.to_string(index=False))

    # Dataset-level kappa win rate.
    c = rec[rec.method == "choice"].set_index(["truth_id", "replicate"])
    j = rec[rec.method == "choice_rt"].set_index(["truth_id", "replicate"])
    common = c.index.intersection(j.index)
    kerr_c = np.abs(c.loc[common, "est_kappa"].to_numpy() - c.loc[common, "true_kappa"].to_numpy())
    kerr_j = np.abs(j.loc[common, "est_kappa"].to_numpy() - j.loc[common, "true_kappa"].to_numpy())
    regimes = c.loc[common, "regime"].to_numpy()

    for reg in ["regular", "weak_beta"]:
        mask = regimes == reg
        if np.any(mask):
            print(f"{reg}: RT gives smaller |kappa error| in "
                  f"{np.mean(kerr_j[mask] < kerr_c[mask]):.1%} of datasets "
                  f"(ties {np.mean(kerr_j[mask] == kerr_c[mask]):.1%}).")

    print("\n=== 6. Plots / package ===")
    for par in ["alpha", "beta", "kappa"]:
        plot_recovery(rec, outdir, par)
    plot_kappa_abs_error(rec, outdir)

    config = {
        "phase": "2b.1_direct_dynamic_recovery",
        "selected_problem_ids": SELECT_IDS,
        "problem_source": source_name,
        "problem_loader": "rfdt_phase2a.load_phase1b_design",
        "phase1b_provenance_audit": "phase1b_provenance_audit.csv",
        "N": N_UNITS,
        "bound": BOUND,
        "r": R_ADAPT,
        "tau_s": TAU_S,
        "t0": T0,
        "max_internal_time": MAX_INTERNAL_TIME,
        "alpha_range": ALPHA_RANGE,
        "beta_range": BETA_RANGE,
        "kappa_range": KAPPA_RANGE,
        "n_atlas": int(args.atlas),
        "n_sim_atlas_per_problem": int(args.sim_atlas),
        "n_rt_bins": N_RT_BINS,
        "n_regular_truths": N_REGULAR_TRUTHS,
        "n_weak_truths": N_WEAK_TRUTHS,
        "n_replicates": N_REPLICATES,
        "n_obs_per_problem": N_OBS_PER_PROBLEM,
        "seeds": {
            "atlas": SEED_ATLAS,
            "bins": SEED_BINS,
            "truths": SEED_TRUTHS,
            "data": SEED_DATA,
        },
        "threads": get_num_threads(),
        "important_interpretation": (
            "Choice-only is the marginal first-passage choice likelihood of the dynamic model. "
            "Choice+RT is the joint choice x fixed-RT-bin first-passage likelihood. "
            "Nuisance parameters are fixed in Phase 2b.1."
        )
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    readme = f"""RFDT v2 — Phase 2b.1 direct dynamic recovery

What this run tests
-------------------
The run compares recovery of alpha, beta, and kappa from the exact finite-N
first-passage model using:

1. choice only:
   the marginal first-passage choice likelihood;

2. choice + RT:
   the joint choice x RT-bin first-passage likelihood.

The two fits use the same five Phase-2a validated stimuli. Stimuli are loaded
through rfdt_phase2a.load_phase1b_design, and their stored nominal Q0/V/QV
values are recomputed before simulation as a provenance check. The accelerated
Numba generator implements the same finite-N heat-bath/Gillespie transition
rates and fixed-N_A rule as the validated Phase-2a generator. The RT likelihood uses fixed, problem-specific bins
constructed from an independent prior-predictive pilot; bin edges are not
adapted to each observed synthetic dataset or candidate theta.

Fixed nuisance parameters
-------------------------
N={N_UNITS}
bound={BOUND}
r={R_ADAPT}
tau_s={TAU_S}
t0={T0}

This is intentional. Phase 2b.1 asks whether RT helps recover the structural
dynamic parameters when the observation scale is known. It does NOT yet claim
that threshold, effective N/noise, internal time scale, and nondecision time
are jointly identifiable.

Interpretation rules
--------------------
- Focus first on recovery_summary.csv and recovery_rmse_gains.csv.
- For kappa, rmse_ratio_choice_over_choice_rt > 1 favors choice+RT.
- Check whether the improvement is broad across regular truth points in
  recovery_by_truth.csv, rather than driven by one truth.
- Treat beta<0.5 as a stress test of the analytically known weak-sensitivity
  singular regime, not as a coding failure.
- Check nearest_atlas_floor.csv before interpreting tiny error differences:
  this pilot uses a finite QMC atlas, so there is a discretization floor.
- Check atlas_diagnostics.csv for timeout rates. Large timeout mass means the
  current parameter domain/max_time needs adjustment before a final run.

What should happen next
-----------------------
If choice+RT materially improves kappa recovery in the regular beta regime,
Phase 2b.2 should free nuisance parameters in stages:
  (i) bound,
  (ii) bound + tau_s + t0,
  (iii) effective N/noise,
and profile the resulting confounds. r should remain fixed at 1 until the data
show a need to free it.

After that, the exact stimulus design of each candidate empirical dataset
should be substituted into the same recovery machinery before any human
responses are fit.
"""
    (outdir / "README_RESULTS.txt").write_text(readme)

    zip_path = Path(str(outdir) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in outdir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(outdir.parent))

    print(f"\nDone. Output directory: {outdir}")
    print(f"ZIP: {zip_path}")
    print("\nUpload the ZIP for the Phase-2b.1 diagnosis.")


if __name__ == "__main__":
    main()
