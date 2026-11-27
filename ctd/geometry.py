"""Analytic model of Stoof & Goni-Moreno (2020): equations 1-3, 7, 8.

Lengths are expressed in residues (AA). With the paper's x_bp = x_AA = 3 A,
two ribosomes that initiated dt apart are separated by
    dx = 3 * v * dt   (in units of 3 A, i.e. "AA contour lengths")
and their nascent chains differ in length by
    dr = v * dt.
"""
from __future__ import annotations

import numpy as np

V = 10.0  # AA/s


def psi(r, dr, dx):
    """Equation 2: relative overlap of spheres with radii r and r+dr at distance dx,
    normalised by the volume of the smaller sphere (radius r)."""
    r, dr, dx = np.broadcast_arrays(np.asarray(r, float), np.asarray(dr, float), np.asarray(dx, float))
    out = np.zeros(r.shape)
    ok = (dr + 2 * r > dx) & (r > 0) & (dx > 0)
    rr, d, x = r[ok], dr[ok], dx[ok]
    val = (d + 2 * rr - x) ** 2 * (x**2 + 2 * x * (2 * rr + d) - 3 * d**2) / (16 * rr**3 * x)
    out[ok] = np.where(x <= d, 1.0, val)  # containment (cannot occur when dx = 3 dr)
    return out


def overlap_integral(dt, R, n_r=400, v=V, dx_factor=3.0):
    """I(dt) = integral_0^{R-dr} Psi(dt, r)^2 dr and the window length R - dr."""
    dt = np.atleast_1d(np.asarray(dt, float))
    dr = v * dt
    win = np.clip(R - dr, 0.0, None)
    u = (np.arange(n_r) + 0.5) / n_r
    r = u[None, :] * win[:, None]
    p = psi(r, dr[:, None], dx_factor * dr[:, None])
    return (p**2).mean(axis=1) * win, win


def alpha_pair(lam, R, kD, form="mm", n_t=600, v=V, dx_factor=3.0):
    """Equation 3: probability that two consecutive nascent chains dimerise
    co-translationally, averaged over the exponential headway distribution.

    form="mm":     paper's saturating form  I / ((R-dr)/kD + I)
    form="hazard": first-passage form       1 - exp(-kD * I / R)
                   (what a stochastic simulation with hazard kc*Psi^2,
                   kc = kD * v / R, produces for an isolated pair)
    """
    if R <= 0:
        return 0.0
    # Psi vanishes for r <= dr, so only headways below R/(2v) contribute.
    tmax = R / (2 * v)
    t = (np.arange(n_t) + 0.5) / n_t * tmax
    I, win = overlap_integral(t, R, v=v, dx_factor=dx_factor)
    if form == "mm":
        frac = np.where(I > 0, I / (win / kD + I), 0.0)
    else:
        frac = 1.0 - np.exp(-kD * I / R)
    w = lam * np.exp(-lam * t)
    return float(np.sum(w * frac) * tmax / n_t)


def alpha_map(lams, lengths, kD, form="mm"):
    return np.array([[alpha_pair(l, R, kD, form) for l in lams] for R in lengths])


def lens_from_caps(r1, r2, d):
    """Independent check of eq. 2 from Methods 4.2 (two spherical caps, eq. 11-13)."""
    xs = (d**2 + r1**2 - r2**2) / (2 * d)
    a2 = r1**2 - xs**2
    h1, h2 = r1 - xs, r2 - d + xs
    cap = lambda h: np.pi * h * (h**2 + 3 * a2) / 6
    return (cap(h1) + cap(h2)) / (4 / 3 * np.pi * min(r1, r2) ** 3)
