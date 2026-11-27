"""Experiments that test the theory against published data.

  1  E. coli DiSP (Bertolini et al. 2021, GEO GSE151959): how much pairing, where it starts
  2  Human DiSP: onset kinetics, loss of pairing towards the stop codon, efficiency vs length
  3  Does the paper's sphere model explain the measured profile? (fit, with and without a gate)
  4  Coupled transcription and translation at the gene: cis, sibling transcripts, territory
  5  Where is the dimer interface of TetR, LacI and lambda cI, and what follows for alpha?

Needs results/geometry.pkl (compute_geometry.py). Writes results/tests.pkl.
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from ctd import disp
from ctd.sim2 import DispParams, Kernel, LocusParams, simulate_disp, simulate_locus
from ctd.structure import read_cif

ROOT = Path(__file__).parent
OUT = ROOT / "results"
T0 = time.time()
NA_A3 = 6.02214e-4


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ experiment 1 and 2: data
def experiment_ecoli():
    d = disp.load_ecoli()
    C = disp.CONSTRUCTS
    out = {}
    mut_m = d["1Bmut_mono_1"] + d["1Bmut_mono_2"]
    reps = {"1B": ("1", "2"), "1Bmut": ("1", "2"), "LMNC": ("1", "2"), "DCTN1": ("1",)}
    for name, rr in reps.items():
        c = C[name]
        for rep in rr:
            M, D = d[f"{name}_mono_{rep}"], d[f"{name}_dis_{rep}"]
            x, E = disp.enrichment(M, D, c["pre"], 10)
            r = dict(x=x, E=E, fit=disp.fit_onset(M, D, hi={"LMNC": 420, "DCTN1": 900}.get(name)),
                     reads=(float(M.sum()), float(D.sum())))
            pl = (x > c["plateau"][0]) & (x < c["plateau"][1])
            r["E_plateau"] = float(np.nanmean(E[pl]))
            if name == "1B":      # reference: the mutant with identical mCherry sequence
                xd, f = disp.depletion_vs_reference(M, mut_m, c["pre"], 10)
                r["f_dep"] = f
                r["f_plateau"] = float(f[pl].mean()); r["f_plateau_se"] = float(f[pl].std(ddof=1) / np.sqrt(pl.sum()))
            elif name == "LMNC":  # no reference construct: assumes even ribosome density along the ORF
                r["f_plateau"] = float(1 - M[c["plateau"][0]:c["plateau"][1]].mean() / M[c["pre"][0]:c["pre"][1]].mean())
            if "f_plateau" in r:
                r["beta"] = (r["f_plateau"] / (1 - r["f_plateau"])) / (r["E_plateau"] - 1)
                r["f"] = disp.f_from_enrichment(E, r["beta"])
            out[(name, rep)] = r
    return out


def experiment_human():
    h = disp.load_human()
    for k in ("mono", "dis", "tot"):
        h[k] = h[k + "1"].astype(np.float64) + h[k + "2"]
    L, on, hi = h["length"], h["onset"], h["high"] == 1
    lowc, ctrl = h["high"] == 0, h["high"] == -1
    out = dict(n_high=int(hi.sum()), onset=on[hi], length=L[hi])
    b = lambda a, w: a[:len(a) // w * w].reshape(-1, w).sum(1)
    # aligned at the onset
    x, o, cov = disp.metagene(h, ["mono", "dis", "tot"], on, -150, 500, hi)
    w = 10
    M, D, T = b(o["mono"], w), b(o["dis"], w), b(o["tot"], w)
    xb = x[:len(M) * w].reshape(-1, w).mean(1)
    pre = (xb > -120) & (xb < -20)
    out["onset_meta"] = dict(x=xb, enrich=(D / M) / (D[pre].sum() / M[pre].sum()),
                             depletion=1 - (M / T) / (M[pre].sum() / T[pre].sum()))
    # aligned at the stop codon: genes with a long stretch after the onset, and a control group
    w = 15
    span = 450
    sets = {"high": hi & ((L - on) >= span), "low": lowc & ((L - on) >= span), "control": ctrl & (L >= span + 100)}
    out["stop_meta"] = {}
    for k, m in sets.items():
        x2, o2, _ = disp.metagene(h, ["mono", "dis", "tot"], L, -span, -3, m)
        M2, D2 = b(o2["mono"], w), b(o2["dis"], w)
        xb2 = x2[:len(M2) * w].reshape(-1, w).mean(1)
        ref = xb2 < -300
        out["stop_meta"][k] = dict(x=xb2, enrich=(D2 / M2) / (D2[ref].sum() / M2[ref].sum()), n=int(m.sum()))
    # bootstrap over genes for the high-confidence stop profile
    idx = np.where(sets["high"])[0]
    rng = np.random.default_rng(0)
    prof = {}
    xr = np.arange(-span, -3)
    per_gene = {k: np.array([h[k][h["offset"][i] + L[i] + xr] for i in idx]) for k in ("mono", "dis")}
    boots = []
    for _ in range(400):
        s_ = rng.integers(0, len(idx), len(idx))
        M2, D2 = b(per_gene["mono"][s_].sum(0), w), b(per_gene["dis"][s_].sum(0), w)
        boots.append((D2 / M2) / (D2[ref].sum() / M2[ref].sum()))
    out["stop_meta"]["high"]["ci"] = np.percentile(boots, [2.5, 97.5], axis=0)
    # per-gene efficiency (definition of Bertolini et al.)
    eff = np.full(len(L), np.nan); nr = np.zeros(len(L))
    for i in np.where(hi)[0]:
        eff[i], nr[i] = disp.gene_efficiency(h, i)
    ok = hi & np.isfinite(eff) & (nr >= 60)
    out["eff"] = dict(eff=eff[ok], remaining=(L - on)[ok], onset=on[ok], length=L[ok], rpkm=h["rpkm"][ok],
                      genes=h["genes"][ok])
    for nm, v in (("remaining", (L - on)[ok]), ("onset", on[ok]), ("length", L[ok]), ("rpkm", h["rpkm"][ok])):
        out["eff"]["rho_" + nm] = stats.spearmanr(v, eff[ok])
    return out


# ------------------------------------------------------------------ experiment 3: model vs measured profile
BIN = 10
FIT_LO, FIT_HI = 20, 340       # the last codons are excluded from the fit (see report)


def _profile(args):
    kind, lam, a, n_on, kern_fit = args
    if kind == "paper":
        k = Kernel(mode="paper", kD=a, tunnel=0, n_on=0)
    elif kind == "gated_paper":
        k = Kernel(mode="paper", kD=a, tunnel=30, n_on=n_on)
    elif kind == "structure":
        k = Kernel(mode="structure", kon=a, n_on=n_on, tunnel=30, **kern_fit)
    else:
        k = Kernel(mode="constant", k_const=a, n_on=n_on, tunnel=30)
    r = simulate_disp(DispParams(L=395, lam=lam, v=15.0, kterm=15.0, T=6000.0, seed=3, kernel=k))
    occ, link = r["occ"][:390], (r["f"] * r["occ"])[:390]
    f = link.reshape(-1, BIN).sum(1) / np.maximum(occ.reshape(-1, BIN).sum(1), 1e-12)
    return f, r["alpha"], r["mean_spacing"]


def experiment_fit(ecoli, kern_fit, ex):
    target = np.mean([ecoli[("1B", r)]["f"][:39] for r in ("1", "2")], axis=0)
    x = (np.arange(39) + 0.5) * BIN
    sel = (x > FIT_LO) & (x < FIT_HI)
    lams = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.5])
    out = dict(x=x, target=target, lams=lams)
    gates = (100, 115, 130)
    grids = {
        "paper": [("paper", l, a, 0, None) for l in lams for a in np.logspace(-1.5, 2.5, 13)],
        "gated_paper": [("gated_paper", l, a, n, None) for l in lams for a in np.logspace(-1, 3, 9) for n in gates],
        "constant": [("constant", l, a, n, None) for l in lams for a in np.logspace(-2, 1.5, 11) for n in gates],
        "structure": [("structure", l, a, n, kern_fit) for l in lams for a in np.logspace(4, 9, 11) for n in gates],
    }
    for name, g in grids.items():
        res = list(ex.map(_profile, g, chunksize=8))
        sse = np.array([np.sum((f[sel] - target[sel]) ** 2) for f, _, _ in res])
        i = int(np.argmin(sse))
        out[name] = dict(f=res[i][0], alpha=res[i][1], spacing=res[i][2], lam=g[i][1], a=g[i][2], n_on=g[i][3],
                         rmse=float(np.sqrt(sse[i] / sel.sum())), sse=sse, grid=[(q[1], q[2], q[3]) for q in g])
        log(f"fit {name}: lam {g[i][1]} a {g[i][2]:.3g} n_on {g[i][3]} rmse {out[name]['rmse']:.3f} alpha {res[i][1]:.2f} spacing {res[i][2]:.0f}")
    return out


# ------------------------------------------------------------------ experiment 4: gene locus
def _locus(p):
    return simulate_locus(p)


def experiment_locus(kern_fit, ex):
    out = {}
    ktx = np.logspace(-2.5, 0, 11)
    base = dict(L=300, lam=0.1, T=4.0e4, mrna_life=180.0)
    for label, n_on in (("N-terminal interface", 100), ("no gate", 0)):
        k = Kernel(mode="structure", kon=1e6, n_on=n_on, tunnel=30, **kern_fit)
        for terr_name, terr in (("DNA-tethered only", 0.0), ("territory 0.2 um", 0.2), ("whole cell", 0.62)):
            ps = [LocusParams(k_tx=float(q), territory_radius=terr, kernel=k,
                              T=float(min(4.0e4, max(1500.0, 400.0 / q))), seed=11, **{a: b for a, b in base.items() if a != "T"})
                  for q in ktx]
            out[(label, terr_name)] = list(ex.map(_locus, ps))
    out["ktx"] = ktx
    # bursty promoter: same mean transcription rate reached by switching a strong promoter on and off
    k = Kernel(mode="structure", kon=1e6, n_on=100, tunnel=30, **kern_fit)
    on_frac = np.array([0.03, 0.06, 0.12, 0.25, 0.5, 1.0])
    ps = [LocusParams(k_tx=0.5, p_on=float(f / (1 - f + 1e-9) / 60.0) if f < 1 else 0.0, p_off=1 / 60.0 if f < 1 else 0.0,
                      territory_radius=0.2, kernel=k, T=float(min(4.0e4, 3000.0 / f)), seed=5, L=300, lam=0.1) for f in on_frac]
    out["bursty"] = dict(on_frac=on_frac, res=list(ex.map(_locus, ps)))
    # polymerase speed: coupled (15 codons/s) versus a polymerase that outruns the ribosomes (T7-like)
    out["speed"] = {v: simulate_locus(LocusParams(k_tx=0.1, v_tx=v, territory_radius=0.0, kernel=k, T=4.0e4, L=300, lam=0.1))
                    for v in (15.0, 75.0)}
    return out


# ------------------------------------------------------------------ experiment 5: repressor interfaces
AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE".split())
REPRESSORS = {"TetR(B)": ("4AC0", 207), "LacI": ("1LBI", 360), "lambda cI": ("3BDN", 237)}


def interface_positions(pid, cutoff=4.0):
    """Residue numbers at the largest protein-protein interface of biological assembly 1."""
    _, at = read_cif(ROOT / "data" / "pdb" / f"{pid}-assembly1.cif.gz", want=())
    prot = np.isin(at["comp"], list(AA)) & (at["element"] != "H")
    chains = list(dict.fromkeys(at["asym"][prot]))
    best = np.array([])
    for i, a in enumerate(chains):
        ma = prot & (at["asym"] == a)
        for b in chains[i + 1:]:
            mb = prot & (at["asym"] == b)
            d, _ = cKDTree(at["xyz"][mb]).query(at["xyz"][ma], distance_upper_bound=cutoff)
            d2, _ = cKDTree(at["xyz"][ma]).query(at["xyz"][mb], distance_upper_bound=cutoff)
            r = np.array(sorted(set(at["seq"][ma][np.isfinite(d)]) | set(at["seq"][mb][np.isfinite(d2)])))
            if len(r) > len(best):
                best = r
    return best


def _alpha(args):
    L, lam, n_on, kern_fit, kon = args
    k = Kernel(mode="structure", kon=kon, n_on=n_on, tunnel=30, **kern_fit)
    return simulate_disp(DispParams(L=L, lam=lam, v=15.0, kterm=15.0, T=max(4.0e4, 4000 / lam), seed=9, kernel=k))["alpha"]


def experiment_repressors(kern_fit, ex):
    out = {}
    for name, (pid, L) in REPRESSORS.items():
        r = interface_positions(pid)
        q = np.percentile(r, [10, 50, 90])
        out[name] = dict(pdb=pid, L=L, residues=r, q10=float(q[0]), q50=float(q[1]), q90=float(q[2]),
                         exposed_before_release=float(np.mean(r <= L - 30)))
    # alpha for a 300-residue protein as a function of where the interface becomes available
    n_on = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 240, 250, 255, 260, 265])
    lams = (0.1, 1.0)
    for kon in (1e6, 1e7):
        res = list(ex.map(_alpha, [(300, l, int(n), kern_fit, kon) for l in lams for n in n_on]))
        out[("scan", kon)] = np.array(res).reshape(len(lams), len(n_on))
    out["scan_n_on"], out["scan_lams"] = n_on, lams
    # the three repressors, lenient criterion: pairing possible once half of the interface residues are exposed
    for name, (pid, L) in REPRESSORS.items():
        n50 = int(out[name]["q50"])
        out[name]["window50"] = L - 30 - n50
        out[name]["window90"] = L - 30 - int(np.ceil(out[name]["q90"]))
        out[name]["alpha"] = {(l, kon): _alpha((L, l, n50, kern_fit, kon)) for l in lams for kon in (1e6, 1e7)}
    return out


def _alpha_lam(args):
    lam, n_on, kon, L = args
    k = Kernel(mode="structure", kon=kon, n_on=n_on, tunnel=30)
    r = simulate_disp(DispParams(L=L, lam=lam, v=15.0, kterm=15.0, T=float(min(2.0e5, max(6000.0, 3000.0 / lam))), seed=4, kernel=k))
    return r["alpha"], r["mean_spacing"]


def experiment_lambda(ex):
    """Does a stronger ribosome binding site always give more co-translational pairing?"""
    from ctd import geometry as g
    lams = np.logspace(-2.3, 0.3, 14)
    out = dict(lams=lams, paper=np.array([g.alpha_pair(l * 10 / 15, 400, 1.0) for l in lams]))  # same spacing at v = 15
    for label, n_on, kon in (("gate 100, kon 1e6", 100, 1e6), ("gate 100, kon 1e5", 100, 1e5), ("no gate, kon 1e6", 0, 1e6)):
        res = list(ex.map(_alpha_lam, [(float(l), n_on, kon, 400) for l in lams]))
        out[label] = dict(alpha=np.array([r[0] for r in res]), spacing=np.array([r[1] for r in res]))
    return out


def main():
    import sys
    only = sys.argv[1:]
    geo = pickle.load(open(OUT / "geometry.pkl", "rb"))
    kern_fit = dict(linker0=float(geo["summary"]["linker_nt"]), s_min=int(geo["s_min"]))
    res = pickle.load(open(OUT / "tests.pkl", "rb")) if only else dict(kern_fit=kern_fit)
    want = lambda k: not only or k in only
    if want("ecoli"):
        res["ecoli"] = experiment_ecoli(); log("E. coli data done")
    if want("human"):
        res["human"] = experiment_human(); log("human data done")
    with ProcessPoolExecutor(max_workers=8) as ex:
        if want("fit"):
            res["fit"] = experiment_fit(res["ecoli"], kern_fit, ex)
        if want("lambda"):
            res["lambda"] = experiment_lambda(ex); log("lambda scan done")
        if want("locus"):
            res["locus"] = experiment_locus(kern_fit, ex); log("locus done")
        if want("repressors"):
            res["repressors"] = experiment_repressors(kern_fit, ex); log("repressors done")
    with open(OUT / "tests.pkl", "wb") as f:
        pickle.dump(res, f)
    log("saved")


if __name__ == "__main__":
    main()
