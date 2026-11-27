"""Three-dimensional simulations of two nascent chains on a tethered pair of ribosomes.

  contacts  contacts between two competent interfaces at fixed chain lengths: time to the first
            contact and fraction of time in contact, for ribosomes of different mobility
            (contacts=16 computes one mobility only)
  lapse     time-lapse trajectories from initiation to the release of both chains; their frames
            go to results/clips/ (not kept in the repository), for paper/render/movie.py

    python3 compute_3d.py [contacts] [lapse]

Writes results/sim3d.pkl (about 90 minutes on 8 cores for both parts).
"""
from __future__ import annotations

import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from ctd import chain3d as c3

OUT = Path(__file__).parent / "results"
T0 = time.time()
MOBILITIES = (1 / 64, 0.25, 1.0, 4.0, 16.0)   # diffusion coefficient of the ribosomes relative to water
SEEDS_CONTACT = range(101, 141)
SEEDS_LAPSE = range(1, 9)
SWEEPS = 12_000_000


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def _contact(args):
    seed, mobility = args
    # above the mobility in water, runs stop at the first contact (their effective concentration is not used)
    return c3.contact_statistics(seed, sweeps=SWEEPS, mobility=mobility, stop_at_contact=mobility > 1.0)


def _lapse(seed):
    steps, meta, movie = c3.timelapse(seed=seed, clips=True)
    (OUT / "clips").mkdir(exist_ok=True)
    with open(OUT / "clips" / f"seed_{seed}.pkl", "wb") as f:
        pickle.dump((steps, meta, movie), f)
    return steps, meta


def rate(runs):
    """First-contact rate (per second) with an exact Poisson interval; runs without contact are censored."""
    n = sum(r["first_contact"] >= 0 for r in runs)
    exposure = sum((r["first_contact"] if r["first_contact"] >= 0 else r["sweeps"]) * r["sweep_time"] for r in runs)
    lo = chi2.ppf(0.025, 2 * n) / 2 / exposure if n else 0.0
    hi = chi2.ppf(0.975, 2 * n + 2) / 2 / exposure
    return dict(events=n, runs=len(runs), exposure=exposure, rate=n / exposure, lo=lo, hi=hi,
                times=np.array([r["first_contact"] * r["sweep_time"] if r["first_contact"] >= 0 else np.nan for r in runs]),
                censor=np.array([r["sweeps"] * r["sweep_time"] for r in runs]),
                mobility=float(np.mean([r["d_ribosome"] for r in runs]) / c3.D_RIBOSOME),
                rotation=float(np.mean([r["d_rot"] for r in runs]) / c3.D_ROT_RIBOSOME))


def power_law(rates):
    """Rate = k1 * mobility^gamma, by maximum likelihood on the numbers of first contacts; the interval of
    gamma is where twice the drop of the log-likelihood stays below 3.84."""
    m = np.array([r["mobility"] for r in rates]); n = np.array([r["events"] for r in rates]); T = np.array([r["exposure"] for r in rates])

    def loglik(g):
        k1 = n.sum() / (m ** g * T).sum()
        k = k1 * m ** g
        return float((n * np.log(k) - k * T).sum()), k1
    best = minimize_scalar(lambda g: -loglik(g)[0], bounds=(-1, 3), method="bounded").x
    top = loglik(best)[0]
    grid = np.linspace(best - 1.5, best + 1.5, 3001)
    ok = np.array([2 * (top - loglik(g)[0]) < 3.84 for g in grid])
    return dict(gamma=float(best), k1=float(loglik(best)[1]), lo=float(grid[ok].min()), hi=float(grid[ok].max()))


def mean_ci(x, rng, n=10_000):
    x = np.asarray(x, float)
    b = rng.choice(x, size=(n, len(x))).mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), runs=len(x))


def analyse(res):
    rng = np.random.default_rng(7)
    st = dict(rate={m: rate(runs) for m, runs in res["contacts"].items()})
    rs = [st["rate"][m] for m in sorted(st["rate"])]
    st["power"] = power_law(rs)
    n, T = sum(r["events"] for r in rs), sum(r["exposure"] for r in rs)
    st["pooled"] = dict(events=n, exposure=T, rate=n / T, lo=chi2.ppf(0.025, 2 * n) / 2 / T, hi=chi2.ppf(0.975, 2 * n + 2) / 2 / T,
                        runs=sum(r["runs"] for r in rs))
    # effective concentration: an equilibrium property, so every mobility gives the same value; runs that were
    # not followed over their whole length are left out
    st["ceff"] = {}
    for m, runs in res["contacts"].items():
        if all(r["sweeps"] >= SWEEPS for r in runs):
            st["ceff"][m] = dict(nterm=mean_ci([r["ceff_nterm"] for r in runs], rng), interface=mean_ci([r["ceff_interface"] for r in runs], rng),
                                 contact_fraction=mean_ci([r["contact_sweeps"] / r["sweeps"] for r in runs], rng))
    allr = [r for m in st["ceff"] for r in res["contacts"][m]]
    st["ceff_all"] = dict(nterm=mean_ci([r["ceff_nterm"] for r in allr], rng), interface=mean_ci([r["ceff_interface"] for r in allr], rng))
    base = res["contacts"][1.0]
    st["sweep_time"] = float(np.mean([r["sweep_time"] for r in base]))
    st["exposed"] = base[0]["exposed"]
    st["spacing"] = base[0]["spacing"]
    lap = res["timelapse"]
    st["lapse_paired"] = sum(m["first_pair_codon"] is not None for _, m in lap)
    st["lapse_runs"] = len(lap)
    st["lapse_first_codons"] = [m["first_pair_codon"] for _, m in lap]
    st["gap_first_pair"] = [m["exit_gap_first_pair"] for _, m in lap if m.get("exit_gap_first_pair") is not None]
    wait = [m["exit_gap_waiting"] for _, m in lap if m.get("exit_gap_waiting")]
    st["gap_waiting"] = np.mean(wait, axis=0).tolist() if wait else None        # mean, 10th, 50th and 90th percentile (A)
    return st


def main():
    parts = sys.argv[1:] or ["contacts", "lapse"]
    path = OUT / "sim3d.pkl"
    res = pickle.load(open(path, "rb")) if path.exists() else dict(contacts={}, timelapse=[])
    with ProcessPoolExecutor(max_workers=8) as ex:
        fut = {}
        if "lapse" in parts:
            res["timelapse"] = []
            fut.update({ex.submit(_lapse, s): ("lapse", s) for s in SEEDS_LAPSE})
        for part in parts:
            if part.startswith("contacts"):
                for m in ([float(part.split("=")[1])] if "=" in part else MOBILITIES):
                    res["contacts"][m] = []
                    fut.update({ex.submit(_contact, (s, m)): ("contact", m) for s in SEEDS_CONTACT})
        for done, f in enumerate(as_completed(fut), 1):
            kind, key = fut[f]
            (res["timelapse"] if kind == "lapse" else res["contacts"][key]).append(f.result())
            if done % 10 == 0:
                log(f"{done} of {len(fut)} runs finished")
    res["timelapse"].sort(key=lambda r: r[1]["seed"])
    for m in res["contacts"]:
        res["contacts"][m].sort(key=lambda r: r["seed"])
    res["stats"] = st = analyse(res)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    for m in sorted(st["rate"]):
        r = st["rate"][m]
        log(f"mobility x{m:g} (realised x{r['mobility']:.2f}): {r['events']} of {r['runs']} runs made contact; "
            f"rate {r['rate']:.0f} per s (95% interval {r['lo']:.0f} to {r['hi']:.0f}); exposure {r['exposure'] * 1e3:.2f} ms")
    p = st["pooled"]
    log(f"all mobilities: {p['events']} contacts in {p['exposure'] * 1e3:.1f} ms: {p['rate']:.0f} per s (95% interval {p['lo']:.0f} to {p['hi']:.0f})")
    log(f"rate = {st['power']['k1']:.0f} per s x mobility^{st['power']['gamma']:.2f} (95% interval of the exponent {st['power']['lo']:.2f} to {st['power']['hi']:.2f})")
    for m in sorted(st["ceff"]):
        c = st["ceff"][m]
        log(f"effective concentration at mobility x{m:g}: N-termini {c['nterm']['mean']:.2f} uM ({c['nterm']['lo']:.2f} to {c['nterm']['hi']:.2f}); "
            f"interface {c['interface']['mean']:.2f} uM ({c['interface']['lo']:.2f} to {c['interface']['hi']:.2f})")
    c = st["ceff_all"]
    log(f"effective concentration, these mobilities together: N-termini {c['nterm']['mean']:.2f} uM ({c['nterm']['lo']:.2f} to {c['nterm']['hi']:.2f}); "
        f"interface {c['interface']['mean']:.2f} uM ({c['interface']['lo']:.2f} to {c['interface']['hi']:.2f})")
    log(f"time-lapses with pairing: {st['lapse_paired']} of {st['lapse_runs']}; trailing ribosome at first pairing: {st['lapse_first_codons']}")
    if st["gap_first_pair"]:
        log(f"distance between the tunnel exits: at first pairing {np.round(st['gap_first_pair']).tolist()} A; while waiting "
            f"mean {st['gap_waiting'][0]:.0f} A (10th to 90th percentile {st['gap_waiting'][1]:.0f} to {st['gap_waiting'][3]:.0f} A)")
    log("saved")


if __name__ == "__main__":
    sys.exit(main())
