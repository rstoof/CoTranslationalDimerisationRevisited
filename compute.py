"""Run every simulation and analysis; store the numbers in results/results.pkl."""
from __future__ import annotations

import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from ctd import chainmc, circuits as c, geometry as g
from ctd.polysome import Params, simulate

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def run(p: Params, target=20000):
    """Simulate long enough to release ~target chains."""
    eff = min(p.lam, p.v / p.footprint)
    p = replace(p, T=target / eff + 500.0, T_burn=500.0)
    r = simulate(p)
    return {k: v for k, v in r.items() if k not in ("snapshots", "rates")}


def _map_row(args):
    R, lams, kD = args
    return [g.alpha_pair(l, R, kD) for l in lams]


def _mc(args):
    n, d, orient, excl = args
    return chainmc.ceff_mc(n, n, d, orient, exclude=excl, n_keep=100_000, seed=n)[0]


VARIANTS = {
    "A": dict(footprint=1, tunnel=0, neighbours_only=True),   # paper limit: point ribosomes, consecutive pairs
    "B": dict(footprint=10, tunnel=0),                          # + ribosome footprint, any pair
    "C": dict(footprint=10, tunnel=30),                         # + exit tunnel hides 30 residues
    "D": dict(footprint=10, tunnel=30, codon_cv=1.0),           # + codon-specific rates (traffic jams)
}


def _sweep(args):
    key, lam, kD = args
    return run(Params(L=300, lam=lam, kD=kD, seed=11, **VARIANTS[key]))


def _design(args):
    lam, R = args
    return run(Params(L=R, lam=lam, kD=1.0, seed=5, **VARIANTS["C"]), target=12000)["alpha"]


def _poly(args):
    lam, kon = args
    return run(Params(L=300, lam=lam, mode="polymer", kon=kon, tunnel=30, footprint=10,
                      x_codon=17.7, seed=3), target=12000)["alpha"]


def _tog(args):
    a, d, lit = args
    t, y = c.toggle_simulate(a, d, literal=lit)
    return min(c.toggle_score(t, y)["ratios"])


def main():
    res = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        # 1 -- reproduction of Figure 2 (equation 3)
        lams = np.logspace(-3, 1, 41)
        lengths = np.linspace(10, 400, 40)
        res["fig2"] = dict(lams=lams, lengths=lengths)
        for kD in (1.0, 10.0):
            res["fig2"][kD] = np.array(list(ex.map(_map_row, [(R, lams, kD) for R in lengths])))
        log("fig2 maps done")

        # 2 -- stochastic polysome vs. analytic pair probability, R = 300 AA
        sl = np.logspace(-3, 1, 13)
        sw = dict(lams=sl)
        for kD in (1.0, 10.0):
            sw[("mm", kD)] = np.array([g.alpha_pair(l, 300, kD, "mm") for l in sl])
            sw[("hazard", kD)] = np.array([g.alpha_pair(l, 300, kD, "hazard") for l in sl])
            for key in VARIANTS:
                sw[(key, kD)] = list(ex.map(_sweep, [(key, l, kD) for l in sl]))
        res["sweep"] = sw
        log("lambda sweep done")

        # 3 -- kymographs
        res["kymo"] = {}
        for lam in (0.1, 1.0):
            p = Params(L=300, lam=lam, kD=1.0, seed=2, snap_dt=0.25, max_snap=480, T=800, T_burn=300,
                       **VARIANTS["B"])
            res["kymo"][lam] = simulate(p)["snapshots"]

        # 4 -- design map (lambda x protein length), realistic variant C
        dl = np.logspace(-3, 1, 17)
        dR = np.array([60, 100, 150, 200, 250, 300, 350, 400])
        vals = list(ex.map(_design, [(l, int(R)) for R in dR for l in dl]))
        res["design"] = dict(lams=dl, lengths=dR, alpha=np.array(vals).reshape(len(dR), len(dl)))
        log("design map done")

        # 5 -- polymer model
        ns = np.array([40, 80, 120, 160, 200, 270])
        pm = dict(ns=ns, gauss=np.array([chainmc.ceff_gauss(n, n, 250.0) for n in ns]))
        for name, orient, excl in (("free", "parallel", False), ("parallel", "parallel", True), ("anti", "anti", True)):
            pm[name] = np.array(list(ex.map(_mc, [(int(n), 250.0, orient, excl) for n in ns])))
        kons = np.logspace(2, 7, 11)
        pm["kons"] = kons
        pm["lams"] = np.array([0.01, 0.1, 1.0])
        pm["alpha"] = np.array(list(ex.map(_poly, [(l, k) for l in pm["lams"] for k in kons]))).reshape(3, -1)
        res["polymer"] = pm
        log("polymer model done")

    # 6 -- repressilator
    al = np.concatenate([np.linspace(0, 0.06, 121), np.linspace(0.065, 1, 60)])
    rep = dict(alphas=al, eig=np.array([c.rep_leading_eig(a) for a in al]),
               alleig=np.array([np.linalg.eigvals(c.rep_jacobian(a)) for a in al]),
               hill=np.array([c.effective_hill(a) for a in al]),
               hopf=c.rep_hopf(), hopf_kun05=c.rep_hopf(kun=0.05))
    rep["tc"] = {a: c.rep_simulate(a, kun=0.05) for a in (0.0, 0.02, 0.1)}
    kd = np.logspace(-10, -4, 25)
    rep["kdim"] = kd
    rep["hopf_kdim"] = np.array([c.rep_hopf(kdim=k) for k in kd])
    ko = np.logspace(-1, 2, 25)
    rep["kon"] = ko
    rep["hopf_kon"] = np.array([c.rep_hopf(kon=k) for k in ko])
    res["rep"] = rep
    log("repressilator done", rep["hopf"], rep["hopf_kun05"])

    # 7 -- toggle switch
    tog = {}
    for lit in (True, False):
        for d in (2.0, 3.0, 7.0):
            for a in (0.02, 0.1, 0.5):
                t, y = c.toggle_simulate(a, d, literal=lit)
                tog[(lit, d, a)] = dict(t=t, G1=y[8], G2=y[9], A1=y[0], A2=y[1], **c.toggle_score(t, y))
    res["toggle"] = tog
    ta = np.array([0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7])
    td = np.linspace(2, 7, 11)
    with ProcessPoolExecutor(max_workers=8) as ex:
        for lit in (True, False):
            sc = list(ex.map(_tog, [(a, d, lit) for d in td for a in ta]))
            res[("toggle_scan", lit)] = dict(alphas=ta, ds=td,
                                            ratio=np.array(sc).reshape(len(td), len(ta)))
    log("toggle done")

    with open(OUT / "results.pkl", "wb") as f:
        pickle.dump(res, f)
    log("saved")


if __name__ == "__main__":
    sys.exit(main())
