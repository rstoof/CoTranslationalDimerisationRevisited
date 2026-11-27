"""Effective concentration between two nascent chains as a function of both chain lengths and
of the distance between the ribosomes along the mRNA, from the atomic ribosome-pair structure.

The average over ribosome orientations is dominated by rare arrangements in which the two exit
tunnels happen to face each other, so it needs many more arrangements than the histogram method of
compute_geometry.py can afford. Here each chain's N-terminal cloud is summarised by its mean and
covariance (measured by Monte Carlo around the real ribosome), the overlap of two clouds is then
analytic, and 20,000 arrangements per spacing are averaged. The shortcut is validated against the
histogram method and against the direct two-ribosome Monte Carlo.

Writes results/kernel.pkl and data/kernel_table.npz.
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ctd import pairgeom as pg

ROOT = Path(__file__).parent
OUT = ROOT / "results"
T0 = time.time()
N_GRID = np.array([20, 40, 60, 80, 100, 130, 160, 200, 250, 300, 400, 550])
EXTRAS = np.array([2, 5, 10, 20, 40, 80, 160, 320, 640])
N_PLACE = 20_000
NA_A3 = 6.02214e-4
_DS = None


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def ds():
    global _DS
    if _DS is None:
        _DS = pg.Disome()
    return _DS


def _moments(n):
    d = ds()
    e, acc = pg.grow_chains(d.grid1.clearance, d.exit1, int(n), 60_000, np.random.default_rng(int(n)))
    return int(n), e.mean(0), np.cov(e.T), e.astype(np.float32)


def gauss_overlap(d, mom, R, t, na, nb, block=True):
    """Effective concentration (M) for a leading chain of `na` residues on ribosome 1 and a trailing
    chain of `nb` residues on ribosome 2, for arrangements (R, t) of ribosome 2. Vectorised."""
    mu_a, S_a = mom[int(na)][0], mom[int(na)][1]
    mu_b0 = d.R12 @ mom[int(nb)][0] + d.t12
    S_b0 = d.R12 @ mom[int(nb)][1] @ d.R12.T
    mu_b = np.einsum("pij,j->pi", R, mu_b0) + t
    S_b = np.einsum("pij,jk,plk->pil", R, S_b0, R)
    S = S_a[None] + S_b
    Si = np.linalg.inv(S)
    dl = mu_b - mu_a[None]
    q = np.einsum("pi,pij,pj->p", dl, Si, dl)
    c = np.exp(-0.5 * q) / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(S)) / NA_A3
    if block:  # no contact if the point where the two clouds overlap most lies inside a ribosome
        m = mu_a[None] + np.einsum("ij,pjk,pk->pi", S_a, Si, dl)
        inside1 = d.grid1.clearance(m) < pg.BEAD
        m2 = np.einsum("pji,pj->pi", R, m - t)
        inside2 = d.grid2.clearance(m2) < pg.BEAD
        c = np.where(inside1 | inside2, 0.0, c)
    return c


def _table(extra):
    d = ds()
    mom = _MOM
    rng = np.random.default_rng(500 + int(extra))
    places, acc = pg.tethered_placements(d, 3 * int(extra), N_PLACE, rng, max_trials=5_000_000)
    R = np.array([p[0] for p in places]); t = np.array([p[1] for p in places])
    ex2 = np.einsum("pij,j->pi", R, d.exit2) + t
    u2 = np.einsum("pij,j->pi", R, d.u2)
    e = ex2 - d.exit1
    dist = np.linalg.norm(e, axis=1)
    facing = (np.einsum("i,pi->p", d.u1, e) / dist > 0) & (np.einsum("pi,pi->p", u2, -e) / dist > 0)
    tab = np.zeros((len(N_GRID), len(N_GRID)))
    for i, na in enumerate(N_GRID):
        for j, nb in enumerate(N_GRID):
            tab[i, j] = gauss_overlap(d, mom, R, t, na, nb).mean()
    c100 = gauss_overlap(d, mom, R, t, 100, 100)
    srt = np.sort(c100)[::-1]
    top1 = srt[:max(len(srt) // 100, 1)].sum() / max(srt.sum(), 1e-300)
    return int(extra), dict(table=tab, accept=acc, n_place=len(places), dist=dist, facing=float(facing.mean()),
                            c100=c100, top1_share=float(top1))


def _validate(extra):
    """Histogram overlap of the full clouds (with end-point masking) versus the analytic shortcut."""
    d = ds()
    mom = _MOM
    rng = np.random.default_rng(900 + int(extra))
    places, _ = pg.tethered_placements(d, 3 * int(extra), 250, rng)
    pairs = [(100, 100), (200, 200), (160, 60)]
    hist = np.zeros((len(places), len(pairs))); gau = np.zeros_like(hist)
    R = np.array([p[0] for p in places]); t = np.array([p[1] for p in places])
    for j, (na, nb) in enumerate(pairs):
        gau[:, j] = gauss_overlap(d, mom, R, t, na, nb)
        a_cloud = mom[na][2].astype(np.float64)
        b0 = mom[nb][2].astype(np.float64) @ d.R12.T + d.t12
        for k, (Rk, tk) in enumerate(places):
            b = b0 @ Rk.T + tk
            b = b[d.grid1.clearance(b) >= pg.BEAD]
            a = a_cloud[d.grid2.clearance((a_cloud - tk) @ Rk) >= pg.BEAD]
            hist[k, j] = pg.overlap(a, 1 / len(a_cloud), b, 1 / len(a_cloud))
    return int(extra), dict(pairs=pairs, hist=hist, gauss=gau)


_MOM = None


def _init(mom):
    global _MOM
    _MOM = mom


def main():
    d = ds()
    with ProcessPoolExecutor(max_workers=8) as ex:
        mom = {n: (mu, S, cloud) for n, mu, S, cloud in ex.map(_moments, [int(n) for n in N_GRID])}
    log("chain clouds:", {n: (round(float(np.linalg.norm(mom[n][0] - d.exit1)), 1), round(float(np.sqrt(np.trace(mom[n][1]))), 1)) for n in mom})
    res = dict(n_grid=N_GRID, extras=EXTRAS, moments={n: (mom[n][0], mom[n][1]) for n in mom})

    # the collided arrangement (identity placement): shortcut versus the direct two-ribosome Monte Carlo
    I, z = np.eye(3)[None], np.zeros((1, 3))
    _init(mom)
    col = np.array([[gauss_overlap(d, mom, I, z, na, nb)[0] for nb in N_GRID] for na in N_GRID])
    col_nb = np.array([[gauss_overlap(d, mom, I, z, na, nb, block=False)[0] for nb in N_GRID] for na in N_GRID])
    res["collided"], res["collided_noblock"] = col, col_nb
    geo = pickle.load(open(OUT / "geometry.pkl", "rb"))
    log("collided, shortcut (n, n) nM:", dict(zip(N_GRID.tolist(), np.round(np.diag(col) * 1e9, 3).tolist())))
    log("collided, direct MC  (n, n+8) nM:", dict(zip(geo["n_trail"].tolist(), np.round(geo["direct"] * 1e9, 3).tolist())))

    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(mom,)) as ex:
        val = dict(ex.map(_validate, [5, 40]))
        res["validation"] = val
        for e, v in val.items():
            for j, pr in enumerate(v["pairs"]):
                h, g = v["hist"][:, j], v["gauss"][:, j]
                log(f"validation extra {e} pair {pr}: mean hist {h.mean() * 1e6:.4f} uM, mean shortcut {g.mean() * 1e6:.4f} uM, "
                    f"log10 ratio of means {np.log10(max(g.mean(), 1e-30) / max(h.mean(), 1e-30)):.2f}")
        tabs = dict(ex.map(_table, [int(e) for e in EXTRAS]))
    res["ensemble"] = tabs
    for e in EXTRAS:
        r = tabs[int(e)]
        i100 = list(N_GRID).index(100)
        log(f"extra {int(e):4d}: {r['n_place']} arrangements (accept {r['accept']:.4f}); exits facing {r['facing']:.3f}; "
            f"median exit distance {np.median(r['dist']):.0f} A; c(100,100) = {r['table'][i100, i100] * 1e6:.3f} uM "
            f"(top 1% of arrangements carry {r['top1_share'] * 100:.0f}%); c(n,n) uM: {np.round(np.diag(r['table']) * 1e6, 3).tolist()}")

    # table for the simulator: axis 0 leading chain, axis 1 trailing chain, axis 2 extra codons
    # (extra 0 = collided arrangement; scaled so that its diagonal follows the direct Monte Carlo)
    full = np.zeros((len(N_GRID), len(N_GRID), len(EXTRAS) + 1))
    direct = np.interp(N_GRID, geo["n_trail"], np.log10(np.maximum(geo["direct"], 1e-14)))
    scale = 10 ** np.median(direct[4:] - np.log10(np.maximum(np.diag(col)[4:], 1e-30)))
    res["collided_scale"] = float(scale)
    full[:, :, 0] = col * scale
    for k, e in enumerate(EXTRAS):
        full[:, :, k + 1] = tabs[int(e)]["table"]
    np.savez(ROOT / "data" / "kernel_table.npz", n=N_GRID.astype(float), extra=np.concatenate([[0], EXTRAS]).astype(float),
             log10c=np.log10(np.maximum(full, 1e-16)))
    res["table"] = full
    with open(OUT / "kernel.pkl", "wb") as f:
        pickle.dump(res, f)
    log("saved; collided scale factor", scale)


if __name__ == "__main__":
    main()
