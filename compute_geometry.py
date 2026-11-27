"""Experiment 3: can nascent chains on neighbouring ribosomes reach each other?

Uses the atomic model of two E. coli ribosomes on one mRNA (PDB 9GFT) to compute
  - where the two exit tunnels are and how far apart,
  - the effective concentration between two nascent chains with the real excluded volume,
  - the same for an ensemble of randomly oriented ribosome pairs tethered by a longer mRNA linker.
Writes results/geometry.pkl.
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ctd import pairgeom as pg

OUT = Path(__file__).parent / "results"
T0 = time.time()
N_TRAIL = np.array([40, 60, 80, 100, 120, 150, 200, 250, 300, 400])
EXTRA = np.array([2, 5, 10, 20, 40, 80, 160])          # codons beyond the collided spacing
N_CLOUD = 60_000
N_PLACE = 700
_DS = None


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def ds():
    global _DS
    if _DS is None:
        _DS = pg.Disome()
    return _DS


def _direct(n):
    """Both chains grown in the real two-ribosome environment (collided arrangement)."""
    d = ds()
    rng = np.random.default_rng(1000 + n)
    e1, a1 = pg.grow_chains(d.grid.clearance, d.exit1, n, 400_000, rng)
    e2, a2 = pg.grow_chains(d.grid.clearance, d.exit2, n + d.psite_nt[0] // 3 - d.psite_nt[1] // 3, 400_000, rng)
    return n, pg.overlap(e1, 1 / len(e1), e2, 1 / len(e2)), a1, a2


def _cloud(n):
    """End points of a chain on ribosome 1 that avoids only its own ribosome."""
    d = ds()
    e, acc = pg.grow_chains(d.grid1.clearance, d.exit1, int(n), N_CLOUD, np.random.default_rng(int(n)))
    return int(n), e.astype(np.float32)


def _ensemble(args):
    extra, clouds, s_min = args
    d = ds()
    rng = np.random.default_rng(77 + int(extra))
    places, accept = pg.tethered_placements(d, 3 * int(extra), N_PLACE, rng)
    dist = np.zeros(len(places)); ang1 = np.zeros(len(places)); ang2 = np.zeros(len(places))
    cen = np.zeros(len(places))
    c = np.zeros((len(places), len(N_TRAIL)))
    for k, (R, t) in enumerate(places):
        ex2 = R @ d.exit2 + t
        u2 = R @ d.u2
        e = ex2 - d.exit1
        dist[k] = np.linalg.norm(e)
        ang1[k] = np.degrees(np.arccos(np.clip(d.u1 @ e / dist[k], -1, 1)))
        ang2[k] = np.degrees(np.arccos(np.clip(u2 @ -e / dist[k], -1, 1)))
        cen[k] = np.linalg.norm(R @ d.centre[2] + t - d.centre[1])
        Rt = R @ d.R12                      # ribosome-1 frame -> placed ribosome 2
        tt = R @ d.t12 + t
        for i, n in enumerate(N_TRAIL):
            a = clouds[int(n)].astype(np.float64)                       # trailing chain, on ribosome 2
            b = clouds[int(n + s_min + extra)].astype(np.float64)       # leading chain, on ribosome 1
            a2 = a @ Rt.T + tt
            if np.linalg.norm(a2.mean(0) - b.mean(0)) > 1.0 * (np.ptp(a2, 0).max() + np.ptp(b, 0).max()):
                continue
            a2 = a2[d.grid1.clearance(a2) >= pg.BEAD]                   # trailing chain may not end inside ribosome 1
            binv = (b - t) @ R                                          # leading-chain ends in ribosome 2's own frame
            b = b[d.grid2.clearance(binv) >= pg.BEAD]
            c[k, i] = pg.overlap(a2, 1 / N_CLOUD, b, 1 / N_CLOUD)
    return int(extra), dict(dist=dist, ang1=ang1, ang2=ang2, centre=cen, ceff=c, accept=accept)


def main():
    d = ds()
    res = dict(summary=d.summary(), free_path=d.free_path(), n_trail=N_TRAIL, extra=EXTRA,
               exit1=d.exit1, exit2=d.exit2, u1=d.u1, u2=d.u2, centre=d.centre)
    s_min = (d.psite_nt[0] - d.psite_nt[1]) // 3
    res["s_min"] = s_min
    log("summary", res["summary"], "free path", round(res["free_path"]), "s_min", s_min)

    with ProcessPoolExecutor(max_workers=8) as ex:
        direct = list(ex.map(_direct, [int(n) for n in N_TRAIL]))
        res["direct"] = np.array([r[1] for r in direct])
        res["direct_accept"] = np.array([[r[2], r[3]] for r in direct])
        log("direct MC (collided arrangement), ceff in M:", dict(zip(N_TRAIL.tolist(), res["direct"].tolist())))

        need = sorted({int(n) for n in N_TRAIL} | {int(n + s_min + e) for n in N_TRAIL for e in EXTRA})
        clouds = dict(ex.map(_cloud, need))
        log("clouds", len(clouds))
        ens = dict(ex.map(_ensemble, [(int(e), clouds, s_min) for e in EXTRA]))
        res["ensemble"] = ens
        for e in EXTRA:
            r = ens[int(e)]
            log(f"extra {int(e):3d} codons: {len(r['dist'])} placements (accept {r['accept']:.4f}); exit distance median {np.median(r['dist']):.0f} A "
                f"(5-95%: {np.percentile(r['dist'], 5):.0f}-{np.percentile(r['dist'], 95):.0f}), "
                f"mean ceff (uM) by n: {np.round(r['ceff'].mean(0) * 1e6, 4).tolist()}")

    with open(OUT / "geometry.pkl", "wb") as f:      # save before the optional consistency check
        pickle.dump(res, f)

    # the collided arrangement evaluated with the cloud method (consistency check of the approximation)
    chk = []
    for n in N_TRAIL:
        a = clouds[int(n)].astype(np.float64) @ d.R12.T + d.t12
        key = int(n + s_min)
        if key not in clouds:
            clouds[key] = _cloud(key)[1]
        b = clouds[key].astype(np.float64)
        a = a[d.grid1.clearance(a) >= pg.BEAD]
        b = b[d.grid2.clearance(b) >= pg.BEAD]
        chk.append(pg.overlap(a, 1 / N_CLOUD, b, 1 / N_CLOUD))
    res["collided_cloud"] = np.array(chk)
    log("collided arrangement via cloud method (M):", np.array(chk))

    with open(OUT / "geometry.pkl", "wb") as f:
        pickle.dump(res, f)
    log("saved")


if __name__ == "__main__":
    main()
