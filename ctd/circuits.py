"""Gene-circuit models of the paper with the co-translational fraction alpha.

Repressilator: equations 4-6, parameters from Repressilator.nb.
Toggle switch: equations of ToggleSwitch.nb (eqs2) with local/global dimers.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

# ---------------------------------------------------------------- repressilator
REP = dict(kon=10.0, kdm=1 / 40 / 60, ks=1 / 1200, kun=0.005, kdim=1e-8)


def rep_rhs(t, y, alpha, kon, kdm, ks, kun, kdim):
    p, m, D = y[0:3], y[3:6], y[6:9]
    Dprev = np.roll(D, 1)  # promoter i is repressed by dimer i-1
    dp = kun * (1 - p) - ks * Dprev * p
    dm = (1 - alpha) * kon * p - kdm * m - kdim * m**2
    dD = alpha * kon * p - kdm * D + kdim * m**2
    return np.concatenate([dp, dm, dD])


def rep_steady(alpha, **kw):
    """Symmetric fixed point p1=p2=p3."""
    k = {**REP, **kw}

    def md(p):
        a, b, c = k["kdim"], k["kdm"], -(1 - alpha) * k["kon"] * p
        m = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a) if a > 0 else -c / b
        D = (alpha * k["kon"] * p + k["kdim"] * m**2) / k["kdm"]
        return m, D

    f = lambda p: k["kun"] * (1 - p) - k["ks"] * md(p)[1] * p
    p = brentq(f, 1e-15, 1.0, xtol=1e-16)
    m, D = md(p)
    return p, m, D


def rep_jacobian(alpha, **kw):
    k = {**REP, **kw}
    p, m, D = rep_steady(alpha, **kw)
    J = np.zeros((9, 9))
    for i in range(3):
        j = (i - 1) % 3
        J[i, i] = -k["kun"] - k["ks"] * D
        J[i, 6 + j] = -k["ks"] * p
        J[3 + i, i] = (1 - alpha) * k["kon"]
        J[3 + i, 3 + i] = -k["kdm"] - 2 * k["kdim"] * m
        J[6 + i, i] = alpha * k["kon"]
        J[6 + i, 3 + i] = 2 * k["kdim"] * m
        J[6 + i, 6 + i] = -k["kdm"]
    return J


def rep_leading_eig(alpha, **kw):
    ev = np.linalg.eigvals(rep_jacobian(alpha, **kw))
    return ev[np.argmax(ev.real)]


def rep_hopf(lo=1e-6, hi=0.9, **kw):
    """alpha at which the leading complex pair crosses the imaginary axis."""
    f = lambda a: rep_leading_eig(a, **kw).real
    if f(lo) <= 0:
        return 0.0
    if f(hi) > 0:
        return float("nan")
    return brentq(f, lo, hi, xtol=1e-10)


def rep_simulate(alpha, t_end=5e5, n=6000, **kw):
    k = {**REP, **kw}
    y0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], float)
    t = np.linspace(0, t_end, n)
    sol = solve_ivp(rep_rhs, (0, t_end), y0, t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10,
                    args=(alpha, k["kon"], k["kdm"], k["ks"], k["kun"], k["kdim"]))
    return sol.t, sol.y


def effective_hill(alpha, **kw):
    """Local log-log sensitivity of steady-state dimer level to promoter activity:
    1 = linear (all co-translational), 2 = quadratic (all post-translational, weak dimerisation)."""
    k = {**REP, **kw}
    p, m, D = rep_steady(alpha, **kw)
    dm_dp = (1 - alpha) * k["kon"] / (k["kdm"] + 2 * k["kdim"] * m)
    dD_dp = (alpha * k["kon"] + 2 * k["kdim"] * m * dm_dp) / k["kdm"]
    return dD_dp * p / D


# ---------------------------------------------------------------- toggle switch
def toggle_params(d_log10bp):
    c = 2 * np.sqrt(226644 / 10)
    return dict(aProd=1.2, aUnh=10.0, aloc=1.0 / (10**d_log10bp / c / 60),
                tDeg=20 * np.log(2), ainh=1e7 / c / 60, tdim=100.0, kun=1 / 5, pert=3.0)


def _box(t, t0, w=60.0):
    return 1.0 if abs(t - t0) < w / 2 else 0.0


def toggle_rhs(t, y, alpha, P, literal):
    A1, A2, m1, m2, r1, r2, L1, L2, G1, G2 = y
    if literal:  # exactly as typed in ToggleSwitch.nb (eqs2)
        dA1 = -P["ainh"] * A1 * G2 - P["aloc"] * L1 * A1 + P["aUnh"] * (1 - A1)
        dA2 = -P["ainh"] * A2 * G1 - P["aloc"] * L1 * A1 + P["aUnh"] * (1 - A2)
        dL1 = alpha * P["aProd"] * m1 - P["aloc"] * L1 * A1 - P["kun"] * L1
        dL2 = alpha * P["aProd"] * m2 - P["aloc"] * L2 * A2 - P["kun"] * L2
        dG1 = -G1 / P["tDeg"] + r1**2 / P["tdim"] + P["aloc"] * L1 * A2 + P["kun"] * L1
        dG2 = -G2 / P["tDeg"] + r2**2 / P["tdim"] + P["aloc"] * L2 * A1 + P["kun"] * L2
    else:  # index-consistent reading: local dimers of gene i act on the other promoter
        dA1 = -P["ainh"] * A1 * G2 - P["aloc"] * L2 * A1 + P["aUnh"] * (1 - A1)
        dA2 = -P["ainh"] * A2 * G1 - P["aloc"] * L1 * A2 + P["aUnh"] * (1 - A2)
        dL1 = alpha * P["aProd"] * m1 - P["aloc"] * L1 * A2 - P["kun"] * L1
        dL2 = alpha * P["aProd"] * m2 - P["aloc"] * L2 * A1 - P["kun"] * L2
        dG1 = -G1 / P["tDeg"] + r1**2 / P["tdim"] + P["aloc"] * L1 * A2 + P["kun"] * L1
        dG2 = -G2 / P["tDeg"] + r2**2 / P["tdim"] + P["aloc"] * L2 * A1 + P["kun"] * L2
    dm1 = P["aProd"] * A1 - m1 / P["tDeg"]
    dm2 = P["aProd"] * A2 - m2 / P["tDeg"]
    dr1 = P["aProd"] * (1 - alpha) * m1 - r1**2 / P["tdim"] - r1 / P["tDeg"] + P["pert"] * _box(t, 500)
    dr2 = P["aProd"] * (1 - alpha) * m2 - r2**2 / P["tdim"] - r2 / P["tDeg"] + P["pert"] * _box(t, 1500)
    return [dA1, dA2, dm1, dm2, dr1, dr2, dL1, dL2, dG1, dG2]


def toggle_simulate(alpha, d_log10bp, literal=False, t_end=4000.0, **override):
    P = {**toggle_params(d_log10bp), **override}
    y0 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    t = np.linspace(0, t_end, 4001)
    sol = solve_ivp(toggle_rhs, (0, t_end), y0, t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10,
                    args=(alpha, P, literal), max_step=5.0)
    return sol.t, sol.y


def toggle_score(t, y):
    """Switching quality: does the dominant repressor flip after each pulse, and by how much?"""
    G1, G2 = y[8], y[9]
    w = lambda a, b: (t > a) & (t < b)
    s0 = (G2[w(350, 450)].mean(), G1[w(350, 450)].mean())      # before pulse 1: 2 should dominate
    s1 = (G1[w(1300, 1450)].mean(), G2[w(1300, 1450)].mean())  # after pulse 1: 1 should dominate
    s2 = (G2[w(3500, 4000)].mean(), G1[w(3500, 4000)].mean())  # after pulse 2: 2 should dominate
    ratios = [a / max(b, 1e-12) for a, b in (s0, s1, s2)]
    return dict(ratios=ratios, levels=[s0[0], s1[0], s2[0]], bistable=all(r > 2 for r in ratios))
