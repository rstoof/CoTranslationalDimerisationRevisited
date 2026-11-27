"""Bead-level Monte Carlo check of the effective concentration between two
nascent chains, including the excluded volume of both ribosomes.

Each exposed nascent chain is a freely jointed chain whose Kuhn length is set so
that its radius of gyration follows the denatured-protein scaling law
Rg = 1.93 n^0.598 A (Kohn et al. 2004). The chain is tethered at the tunnel exit
on the ribosome surface and may not enter either ribosome.
"""
from __future__ import annotations

import numpy as np

R_RIB = 125.0  # A


def _kuhn(n):
    rg2 = (1.93 * n**0.598) ** 2
    b = 6 * rg2 / (n * 3.8)
    return max(int(round(n * 3.8 / b)), 1), b


def sample_ends(n, tether, centres, n_keep, rng, exclude=True, chunk=40_000):
    """N-terminal positions of FJC chains starting at `tether` that avoid the spheres at `centres`."""
    nk, b = _kuhn(n)
    out, tried = [], 0
    while sum(len(o) for o in out) < n_keep and tried < 60 * n_keep:
        v = rng.normal(size=(chunk, nk, 3))
        v *= b / np.linalg.norm(v, axis=2, keepdims=True)
        pts = tether + np.cumsum(v, axis=1)
        ok = np.ones(chunk, bool)
        if exclude:
            for c in centres:
                ok &= (np.sum((pts - c) ** 2, axis=2) > R_RIB**2).all(axis=1)
        out.append(pts[ok, -1, :])
        tried += chunk
    return np.concatenate(out)[:n_keep], sum(len(o) for o in out) / tried


def ceff_mc(n1, n2, d, orient="parallel", exclude=True, n_keep=300_000, seed=0, bin_w=20.0):
    """Effective concentration (M) from the overlap of the two N-terminal densities."""
    rng = np.random.default_rng(seed)
    c1, c2 = np.zeros(3), np.array([d, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    t1 = c1 + R_RIB * up
    t2 = c2 + R_RIB * (up if orient == "parallel" else -up)
    e1, acc1 = sample_ends(n1, t1, (c1, c2), n_keep, rng, exclude)
    e2, acc2 = sample_ends(n2, t2, (c1, c2), n_keep, rng, exclude)
    lo = np.minimum(e1.min(0), e2.min(0)); hi = np.maximum(e1.max(0), e2.max(0))
    edges = [np.arange(l, h + bin_w, bin_w) for l, h in zip(lo, hi)]
    h1, _ = np.histogramdd(e1, edges); h2, _ = np.histogramdd(e2, edges)
    dv = bin_w**3
    p = np.sum((h1 / len(e1) / dv) * (h2 / len(e2) / dv)) * dv  # per A^3
    return p / 6.02214e-4, (acc1, acc2)


def ceff_gauss(n1, n2, d):
    rg1, rg2 = 1.93 * n1**0.598, 1.93 * n2**0.598
    s2 = 2 * (rg1**2 + rg2**2)
    return (2 * np.pi * s2) ** -1.5 * np.exp(-d * d / (2 * s2)) / 6.02214e-4
