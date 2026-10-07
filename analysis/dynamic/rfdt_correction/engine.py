"""Inherited RFDT engine; functions below are extracted verbatim from the validated source.
Only a minimal import header was added. See legacy/ENGINE_PROVENANCE.json.
No transition, initialization, rounding or stopping calculation was rewritten.
The caller now passes a longer computational horizon and rejects unresolved cells.
"""
import math
import numpy as np
from numba import njit

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

