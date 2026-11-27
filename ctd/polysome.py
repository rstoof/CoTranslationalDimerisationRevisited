"""Python wrapper around the Gillespie polysome simulation in polysome.c."""
from __future__ import annotations

import ctypes
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = None

# Physical constants used by the paper (Methods 4.1)
V_TRANSLATE = 10.0   # AA / s
X_BP = 3.0           # Angstrom per nucleotide (paper's value)
X_AA = 3.0           # Angstrom contour length per residue (paper's value)


def _lib():
    global _LIB
    if _LIB is None:
        so, src = _HERE / "libpolysome.so", _HERE / "polysome.c"
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            arch = ["-arch", platform.machine()] if sys.platform == "darwin" else []   # match the interpreter (Rosetta or native)
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", *arch, "-o", str(so), str(src), "-lm"], check=True)
        _LIB = ctypes.CDLL(str(so))
        dp = ctypes.POINTER(ctypes.c_double)
        _LIB.simulate.argtypes = [dp] * 6
        _LIB.simulate.restype = ctypes.c_int
        _LIB.psi_spheres.argtypes = [ctypes.c_double] * 3
        _LIB.psi_spheres.restype = ctypes.c_double
        _LIB.ceff_polymer.argtypes = [ctypes.c_double] * 3
        _LIB.ceff_polymer.restype = ctypes.c_double
    return _LIB


@dataclass
class Params:
    L: int = 300                 # protein length (AA)
    lam: float = 0.1             # ribosome initiation rate (1/s)
    footprint: int = 10          # codons occupied by one ribosome (1 = point particles, as in the paper)
    mode: str = "spheres"        # "spheres" (paper) or "polymer"
    kD: float = 1.0              # dimensionless reaction parameter (spheres mode)
    kon: float = 1e6             # association rate constant, 1/M/s (polymer mode)
    tunnel: int = 0              # residues hidden in the exit tunnel (paper: 0; real ribosome: ~30)
    x_aa: float = X_AA
    x_codon: float = 3 * X_BP
    d_rib: float = 250.0         # ribosome diameter (A), polymer mode
    b_rna: float = 20.0          # Kuhn length of single-stranded mRNA (A), polymer mode
    n_dom: int = 50              # residues that must be exposed before the interface exists
    v: float = V_TRANSLATE       # mean elongation rate (AA/s)
    codon_cv: float = 0.0        # coefficient of variation of codon-specific rates (gamma distributed)
    T: float = 2.0e5             # simulated seconds
    T_burn: float = 200.0
    seed: int = 1
    neighbours_only: bool = False
    snap_dt: float = 0.0
    max_snap: int = 0
    rates: np.ndarray | None = field(default=None, repr=False)

    @property
    def kc(self) -> float:
        """Hazard prefactor (1/s) such that kD = kc * (translation time)."""
        return self.kD * self.v / self.L


def simulate(p: Params) -> dict:
    lib = _lib()
    rng = np.random.default_rng(p.seed + 7919)
    if p.rates is not None:
        rate = np.asarray(p.rates, dtype=np.float64)
    elif p.codon_cv > 0:
        k = 1.0 / p.codon_cv**2
        # gamma-distributed dwell times with mean 1/v -> rates are their inverse
        dwell = rng.gamma(k, 1.0 / (k * p.v), size=p.L)
        rate = 1.0 / dwell
    else:
        rate = np.full(p.L, p.v)
    rate = np.ascontiguousarray(rate)
    mode = 0 if p.mode == "spheres" else 1
    par = np.array([p.L, p.lam, p.footprint, mode, p.kc if mode == 0 else p.kon, p.tunnel,
                    p.x_aa, p.x_codon, p.d_rib, p.b_rna, p.n_dom, p.T, p.seed, p.v,
                    p.snap_dt if p.snap_dt > 0 else 1.0, p.max_snap, p.T_burn,
                    1.0 if p.neighbours_only else 0.0], dtype=np.float64)
    out = np.zeros(8)
    hl, hs = np.zeros(p.L + 1), np.zeros(p.L + 1)
    snap = np.zeros(max(p.max_snap, 1) * 128)
    dp = ctypes.POINTER(ctypes.c_double)
    ns = lib.simulate(*(a.ctypes.data_as(dp) for a in (par, rate, out, hl, hs, snap)))
    n_co, n_free = out[0], out[1]
    n = n_co + n_free
    alpha = n_co / n if n else float("nan")
    return dict(
        alpha=alpha,
        alpha_se=float(np.sqrt(max(alpha * (1 - alpha), 1e-12) / max(n / 2, 1))) if n else float("nan"),
        n_released=int(n), n_co=int(n_co), n_free=int(n_free),
        init_rate=out[2] / out[4], mean_ribosomes=out[3] / out[4],
        blocked_fraction=out[5] / out[3] if out[3] else 0.0,
        dimer_events=int(out[6]), mean_dwell=out[7] / n if n else float("nan"),
        hist_len=hl, hist_sep=hs,
        snapshots=snap.reshape(-1, 64, 2)[:ns] if p.max_snap else None,
        rates=rate,
    )


def psi_c(r1, r2, d):
    return _lib().psi_spheres(float(r1), float(r2), float(d))


def ceff_c(n1, n2, dc):
    return _lib().ceff_polymer(float(n1), float(n2), float(dc))
