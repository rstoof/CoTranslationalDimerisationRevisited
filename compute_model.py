"""Validation of the competence-window model (ctd/window.py) against stochastic polysome simulations.

  A  constant pairing rate, point ribosomes: tests the chain-accounting argument (alpha = 2q/(1+q))
  B  constant pairing rate, ribosome footprint of 10 codons: tests the minimal-headway correction
  C  structure-based pairing rate that depends on the distance between the ribosomes

Writes results/model.pkl.
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ctd import window as w
from ctd.sim2 import DispParams, Kernel, ceff, simulate_disp

OUT = Path(__file__).parent / "results"
T0 = time.time()
L, V, TUNNEL = 400, 15.0, 30
LAMS = np.logspace(-2.5, 0, 11)
WINDOWS = np.array([15, 45, 120, 270])          # codons between interface emergence and stop
KS = np.array([0.03, 0.1, 0.3, 1.0, 10.0])      # pairing rates (1/s)
KONS = np.array([1e5, 1e6, 1e7])
FOOT = 10                                        # ribosome footprint (codons)
S_MIN = 8                                        # codons between two ribosomes in the collided structure


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def _run(args):
    kind, lam, win, a, fp = args
    n_on = L - TUNNEL - int(win)
    if kind == "constant":
        k = Kernel(mode="constant", k_const=a, n_on=n_on, tunnel=TUNNEL)
    else:
        k = Kernel(mode="structure", kon=a, n_on=n_on, tunnel=TUNNEL)
    target = 3.0e4 if lam < 0.3 else 1.2e4
    r = simulate_disp(DispParams(L=L, lam=lam, footprint=fp, v=V, kterm=V, T=target / min(lam, 0.6) + 300.0, T_burn=300.0,
                                 seed=21, kernel=k))
    return r["alpha"], r["init_rate"], r["n_released"]


def rate_from_structure(kon, win):
    """Pairing rate for two ribosomes s codons apart, with chain lengths taken at mid-overlap."""
    n_on = L - TUNNEL - int(win)

    def kfun(s, t_overlap):
        nt = n_on + t_overlap * V / 2
        return kon * ceff(nt + s, nt, max(s - S_MIN, 0.0))
    return kfun


def stats(sim, pred):
    e = sim - pred
    return dict(rms=float(np.sqrt(np.mean(e ** 2))), max=float(np.abs(e).max()), mean=float(e.mean()))


def main():
    res = dict(L=L, v=V, tunnel=TUNNEL, lams=LAMS, windows=WINDOWS, ks=KS, kons=KONS, footprint_codons=FOOT)
    grid_c = [(lam, win, k) for k in KS for win in WINDOWS for lam in LAMS]
    grid_s = [(lam, win, k) for k in KONS for win in WINDOWS for lam in LAMS]
    Wsec = ((WINDOWS + 1) / V)[None, :, None]                # +1: dwell on the stop codon
    with ProcessPoolExecutor(max_workers=8) as ex:
        for name, fp in (("point", 1), ("footprint", FOOT)):
            out = list(ex.map(_run, [("constant", l, wn, k, fp) for l, wn, k in grid_c], chunksize=4))
            a = np.array([o[0] for o in out]).reshape(len(KS), len(WINDOWS), len(LAMS))
            lam_eff = np.array([o[1] for o in out]).reshape(a.shape)
            pred = w.alpha(lam_eff, Wsec, KS[:, None, None], tau_r=0.0 if fp == 1 else fp / V)
            res[name] = dict(alpha=a, lam_eff=lam_eff, pred=pred, stats=stats(a, pred))
            if fp > 1:
                res[name]["stats_uncorrected"] = stats(a, w.alpha(lam_eff, Wsec, KS[:, None, None]))
            log(f"constant rate, {name}:", res[name]["stats"])
        out = list(ex.map(_run, [("structure", l, wn, k, FOOT) for l, wn, k in grid_s], chunksize=4))
    a = np.array([o[0] for o in out]).reshape(len(KONS), len(WINDOWS), len(LAMS))
    lam_eff = np.array([o[1] for o in out]).reshape(a.shape)
    pred = np.array([[[w.alpha_kernel(lam_eff[i, j, l], (wn + 1) / V, rate_from_structure(kon, wn), V, tau_r=FOOT / V)
                       for l in range(len(LAMS))] for j, wn in enumerate(WINDOWS)] for i, kon in enumerate(KONS)])
    const = w.alpha(lam_eff, Wsec, KONS[:, None, None] * 1e-6, tau_r=FOOT / V)       # constant c_eff = 1 uM
    long = WINDOWS >= 120
    res["structure"] = dict(alpha=a, lam_eff=lam_eff, pred=pred, pred_const=const, stats=stats(a, pred),
                            stats_const=stats(a, const), stats_const_long=stats(a[:, long], const[:, long]))
    log("structure-based rate, distance-dependent model:", res["structure"]["stats"])
    log("structure-based rate, constant 1 uM:", res["structure"]["stats_const"], "| windows >= 120 codons:", res["structure"]["stats_const_long"])

    # a convenient description of the structure-based effective concentration (chains of 100 exposed residues)
    kt = np.load(Path(__file__).parent / "data" / "kernel_table.npz")
    i100 = list(kt["n"]).index(100.0)
    s = kt["extra"][1:] + S_MIN
    c = 10 ** kt["log10c"][i100, i100, 1:]
    hill = lambda p, x: p[0] / (1 + (p[1] / x) ** p[2])
    fit = least_squares(lambda p: np.log(hill(p, s[:7])) - np.log(c[:7]), [2e-6, 80.0, 2.0], bounds=([1e-8, 5, 0.5], [1e-4, 500, 8]))
    res["ceff_fit"] = dict(s=s, c=c, c_inf=float(fit.x[0]), s0=float(fit.x[1]), m=float(fit.x[2]),
                           rms_log10=float(np.sqrt(np.mean(fit.fun ** 2)) / np.log(10)))
    log("c_eff(s) fit:", {k: (round(v, 9) if k == "c_inf" else round(v, 2)) for k, v in res["ceff_fit"].items() if np.isscalar(v)})
    with open(OUT / "model.pkl", "wb") as f:
        pickle.dump(res, f)
    log("saved")


if __name__ == "__main__":
    main()
