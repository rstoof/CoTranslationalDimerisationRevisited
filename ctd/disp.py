"""Re-analysis of the Disome Selective Profiling (DiSP) data of Bertolini et al. (Science 2021).

A ribosome whose nascent chain is bound to the nascent chain of another ribosome sediments as a
disome after RNase digestion. With f(x) the fraction of ribosomes at codon x that are linked,
b the fraction of unlinked ribosomes that still end up in the disome fraction, and rho(x) the
ribosome density:
    mono(x) ~ rho(x) (1 - f) (1 - b)          dis(x) ~ rho(x) [f + (1 - f) b]
so that the enrichment E = (dis/mono) / (dis/mono before onset) obeys   E - 1 = f / ((1 - f) b).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

DATA = Path(__file__).resolve().parent.parent / "data"
TUNNEL = 30  # residues hidden in the exit tunnel

# construct coordinates (codons, 1-based) from Table S5 of Bertolini et al.
CONSTRUCTS = {
    "1B": dict(length=395, label="coil 1B–mCherry", pre=(20, 120), plateau=(240, 340),
               features=[("coil 1B", 11, 148), ("mCherry", 155, 390)]),
    "1Bmut": dict(length=395, label="coil 1B* (a↔e swap)–mCherry", pre=(20, 120), plateau=(240, 340),
                  features=[("coil 1B*", 11, 148), ("mCherry", 155, 390)]),
    # lamin C residue = construct codon + 22
    "LMNC": dict(length=587, label="lamin C (31–572)", pre=(20, 120), plateau=(240, 360),
                 features=[("1A", 12, 48), ("coil 1B", 59, 196), ("coil 2", 221, 361), ("staart", 362, 550)]),
    "DCTN1": dict(length=1325, label="p150glued (DCTN1)", pre=(20, 400), plateau=(620, 1000),
                  features=[("CAP", 37, 123), ("coiled coil 1", 222, 555), ("CC2", 951, 1057)]),
}


def load_ecoli():
    return np.load(DATA / "disp_ecoli.npz")


def binned(a, width, lo=0, hi=None):
    hi = len(a) if hi is None else hi
    n = (hi - lo) // width
    return a[lo:lo + n * width].reshape(n, width).sum(1), lo + (np.arange(n) + 0.5) * width


def enrichment(mono, dis, pre, width=10):
    """Disome/monosome ratio per bin, relative to the ratio before onset."""
    m, x = binned(mono, width)
    d, _ = binned(dis, width)
    base = dis[pre[0]:pre[1]].sum() / mono[pre[0]:pre[1]].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return x, (d / m) / base


def f_from_enrichment(E, beta):
    o = beta * np.clip(E - 1, 0, None)
    return o / (1 + o)


def depletion_vs_reference(mono, ref, pre, width=10):
    """f(x) = 1 - [mono/ref](x) / [mono/ref](pre-onset): fraction of ribosomes missing from the
    monosome fraction, using a construct of identical sequence that does not assemble as reference."""
    m, x = binned(mono, width)
    r, _ = binned(ref, width)
    k = mono[pre[0]:pre[1]].sum() / ref[pre[0]:pre[1]].sum()
    return x, 1 - (m / r) / k


def sigmoid(x, lo, hi, a, mid):
    return lo + (hi - lo) / (1 + np.exp(-a * (x - mid)))


def fit_onset(mono, dis, width=5, lo=20, hi=None):
    """Single-sigmoid fit to the disome share p = dis/(dis+mono), as in the original analysis.
    Returns dict(mid, onset10, width, p_before, p_after)."""
    m, x = binned(mono, width, lo, hi)
    d, _ = binned(dis, width, lo, hi)
    ok = (m + d) > 0
    p, w = d[ok] / (m[ok] + d[ok]), np.sqrt((m + d)[ok])
    x = x[ok]
    best = None
    for mid0 in np.linspace(x[2], x[-3], 12):
        try:
            r = least_squares(lambda q: (sigmoid(x, *q) - p) * np.minimum(w, np.median(w)),
                              [np.percentile(p, 10), np.percentile(p, 90), 0.1, mid0],
                              bounds=([0, 0, 0.005, x[0]], [1, 1, 2.0, x[-1]]))
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    lo_, hi_, a, mid = best.x
    return dict(mid=float(mid), onset10=float(mid - np.log(9) / a), width=float(2 * np.log(9) / a),
                p_before=float(lo_), p_after=float(hi_))


# ------------------------------------------------------------------------------ human data
def load_human():
    d = np.load(DATA / "disp_human.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def metagene(h, keys, anchor, lo, hi, genes_mask, min_span=None):
    """Sum of footprints per position relative to `anchor` (array of codon positions per gene)
    for the selected genes. Returns x, {key: summed counts}, number of genes contributing per position."""
    x = np.arange(lo, hi)
    out = {k: np.zeros(len(x)) for k in keys}
    cover = np.zeros(len(x))
    off, L = h["offset"], h["length"]
    for i in np.where(genes_mask)[0]:
        a = int(anchor[i])
        pos = a + x
        ok = (pos >= 3) & (pos < L[i] - 3)
        if min_span is not None and ok.sum() < min_span:
            continue
        idx = off[i] + pos[ok]
        for k in keys:
            out[k][ok] += h[k][idx]
        cover[ok] += 1
    return x, out, cover


def gene_efficiency(h, i, gap=20, edge=8):
    """Co-co assembly efficiency of gene i as defined by Bertolini et al.: reduction of monosome
    footprints after the assembly onset relative to the total translatome. Returns (eff, reads)."""
    off, L, on = h["offset"][i], int(h["length"][i]), int(h["onset"][i])
    b0, b1, a0, a1 = edge, on - gap, on + gap, L - edge
    if b1 - b0 < 20 or a1 - a0 < 20:
        return np.nan, 0
    s = lambda k, lo, hi: float(h[k + "1"][off + lo:off + hi].sum() + h[k + "2"][off + lo:off + hi].sum())
    mb, ma, tb, ta = s("mono", b0, b1), s("mono", a0, a1), s("tot", b0, b1), s("tot", a0, a1)
    if min(mb, tb, ta) < 40 or ma < 5:
        return np.nan, 0
    return 1 - (ma / mb) / (ta / tb), min(mb, ma, tb, ta)
