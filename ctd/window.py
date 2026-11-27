"""Competence-window model of co-translational dimerisation.

Proposed replacement for the overlapping-spheres model of Stoof & Goni-Moreno (2020).

A nascent chain can dimerise only while its dimerisation interface is outside the ribosome:
from the moment the interface has left the exit tunnel until the chain is released. Two chains
on the same mRNA whose competence windows overlap pair with rate k = k_on * c_eff, where the
effective concentration c_eff follows from the structure of the ribosome pair.

    W       = (L - n_i - n_t) / v                 competence window (s)
    Lambda  = lam * integral_0^W [1 - exp(-k(v t) (W - t))] dt
                                                  expected number of following ribosomes that would pair in time
            = lam * (W - (1 - exp(-k W)) / k)     for a constant pairing rate k
    q       = 1 - exp(-Lambda)                    probability that an unpaired competent chain gets a partner
    alpha   = 2 q / (1 + q)                       fraction of chains released in a co-translational dimer
"""
from __future__ import annotations

import numpy as np

N_TUNNEL = 30  # residues held inside the ribosomal exit tunnel


def window(L, n_i, v=10.0, n_tunnel=N_TUNNEL):
    """Seconds between the emergence of the interface and the release of the chain.
    L: protein length (residues); n_i: residues, counted from the N-terminus, that must be
    translated before the interface is complete; v: elongation rate (residues/s)."""
    return np.maximum(np.asarray(L, float) - np.asarray(n_i, float) - n_tunnel, 0.0) / v


def effective_window(W, k=np.inf):
    """Window shortened by the time the reaction itself takes: W - (1 - exp(-k W)) / k."""
    W, k = np.broadcast_arrays(np.asarray(W, float), np.asarray(k, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        x = k * W
        lag = np.where(x < 1e-8, W * (1 - x / 2), -np.expm1(-x) / np.where(k == 0, 1.0, k))
    return np.where(np.isinf(k), W, np.clip(W - lag, 0.0, None))


def _headway(lam, tau_r):
    """Ribosome footprint: minimal headway tau_r, exponential beyond it with the same mean flux."""
    lam = np.asarray(lam, float)
    return lam / np.clip(1.0 - lam * tau_r, 1e-9, None)


def expected_partners(lam, W, k=np.inf, tau_r=0.0):
    """Lambda for a constant pairing rate. tau_r = footprint / v is the minimal headway between
    two ribosomes (0 for point ribosomes)."""
    Wr = np.clip(np.asarray(W, float) - tau_r, 0.0, None)
    return _headway(lam, tau_r) * effective_window(Wr, k)


def alpha(lam, W, k=np.inf, tau_r=0.0):
    """Fraction of chains that leave the ribosome as part of a co-translational dimer."""
    q = -np.expm1(-expected_partners(lam, W, k, tau_r))
    return 2.0 * q / (1.0 + q)


def alpha_kernel(lam, W, kfun, v, tau_r=0.0, n=400):
    """Same, for a pairing rate that depends on the distance between the two ribosomes:
    kfun(s, t_overlap) returns the rate (1/s) for ribosomes s codons apart."""
    if W <= tau_r:
        return 0.0
    t = tau_r + (np.arange(n) + 0.5) / n * (W - tau_r)
    p = 1.0 - np.exp(-np.array([kfun(v * ti, W - ti) for ti in t]) * (W - t))
    Lam = float(_headway(lam, tau_r)) * p.mean() * (W - tau_r)
    q = -np.expm1(-Lam)
    return 2.0 * q / (1.0 + q)


def critical_window(lam, k, alpha_c, tau_r=0.0, w_max=1e4):
    """Longest competence window (s) for which alpha stays below alpha_c."""
    from scipy.optimize import brentq
    f = lambda w: float(alpha(lam, w, k, tau_r)) - alpha_c
    if f(w_max) < 0:
        return np.inf
    return brentq(f, 0.0, w_max, xtol=1e-9)
