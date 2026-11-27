"""Python wrappers for polysome2.c (DiSP observable and gene-locus simulation)."""
from __future__ import annotations

import ctypes
import platform
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = None
DP = ctypes.POINTER(ctypes.c_double)


def _lib():
    global _LIB
    if _LIB is None:
        so, src = _HERE / "libpolysome2.so", _HERE / "polysome2.c"
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            arch = ["-arch", platform.machine()] if sys.platform == "darwin" else []   # match the interpreter (Rosetta or native)
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", *arch, "-o", str(so), str(src), "-lm"], check=True)
        _LIB = ctypes.CDLL(str(so))
        _LIB.simulate_disp.argtypes = [DP] * 7
        _LIB.simulate_locus.argtypes = [DP] * 3
        _LIB.ceff_table.argtypes = [ctypes.c_double] * 3
        _LIB.ceff_table.restype = ctypes.c_double
        _LIB.set_table.argtypes = [DP, ctypes.c_int, DP, ctypes.c_int, DP]
        tab = _HERE.parent / "data" / "kernel_table.npz"
        if tab.exists():
            t = np.load(tab)
            n, e, c = (np.ascontiguousarray(t[k], dtype=np.float64) for k in ("n", "extra", "log10c"))
            _LIB.set_table(n.ctypes.data_as(DP), len(n), e.ctypes.data_as(DP), len(e), c.ctypes.data_as(DP))
    return _LIB


@dataclass
class Kernel:
    """How two nascent chains find each other."""
    mode: str = "structure"     # "paper" | "structure" | "constant"
    kD: float = 1.0             # paper: dimensionless reaction parameter (kc = kD * v / L)
    kon: float = 1e6            # structure: association rate constant (1/M/s)
    A: float = 1.0              # unused
    rho_exit: float = 0.0       # unused
    d0: float = 0.0             # unused
    linker0: float = 11.0       # nucleotides between two ribosomes at the collided spacing
    nt_len: float = 5.9
    b_rna: float = 20.0
    s_min: int = 8              # codons between two ribosomes at the collided spacing
    n_on: int = 0               # exposed residues required on both chains (interface exposure)
    tunnel: int = 30
    k_const: float = 1.0        # constant: pairing rate (1/s)
    s_max: int = 100000
    x_aa: float = 3.0
    x_codon: float = 9.0
    rg0: float = 1.93
    rg_exp: float = 0.598

    def array(self, v=10.0, L=300):
        mode = {"paper": 0, "structure": 2, "constant": 3}[self.mode]
        return np.array([mode, self.kD * v / L, self.x_aa, self.x_codon, self.kon, self.A, self.rho_exit,
                         self.d0, self.linker0, self.nt_len, self.b_rna, self.s_min, self.n_on, self.tunnel,
                         self.k_const, self.s_max, self.rg0, self.rg_exp], dtype=np.float64)



def ceff(n_lead, n_trail, extra_codons=0.0):
    """Effective concentration (M) from the structure-based table."""
    return _lib().ceff_table(float(n_lead), float(n_trail), float(extra_codons))


@dataclass
class DispParams:
    L: int = 395
    lam: float = 0.3
    footprint: int = 10
    v: float = 15.0
    kterm: float = 15.0
    T: float = 2.0e4
    T_burn: float = 300.0
    seed: int = 1
    neighbours_only: bool = False
    kernel: Kernel = field(default_factory=Kernel)


def simulate_disp(p: DispParams) -> dict:
    lib = _lib()
    par = np.array([p.L, p.lam, p.footprint, p.v, p.kterm, p.T, p.T_burn, p.seed,
                    1.0 if p.neighbours_only else 0.0], dtype=np.float64)
    kern = p.kernel.array(p.v, p.L)
    out = np.zeros(6)
    arrs = [np.zeros(p.L + 1) for _ in range(4)]
    lib.simulate_disp(*(a.ctypes.data_as(DP) for a in (par, kern, out, *arrs)))
    occ_all, occ_link, hist_sep, hist_pos = arrs
    n = out[0] + out[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(occ_all > 0, occ_link / occ_all, 0.0)
    return dict(alpha=out[0] / n if n else np.nan, n_released=int(n), f=f, occ=occ_all,
                init_rate=out[2] / out[4], mean_ribosomes=out[3] / out[4],
                mean_spacing=p.L / max(out[3] / out[4], 1e-9), pair_events=int(out[5]),
                hist_sep=hist_sep, hist_pos=hist_pos)


@dataclass
class LocusParams:
    L: int = 300                 # codons
    k_tx: float = 0.05           # transcription initiation (1/s) while the promoter is on
    v_tx: float = 15.0           # polymerase speed (codons/s): 45 nt/s for E. coli RNAP, ~75 for T7
    lam: float = 0.1             # ribosome initiation per transcript (1/s)
    v_rib: float = 15.0          # ribosome speed (codons/s)
    kterm: float = 15.0
    footprint: int = 10
    rnap_footprint: int = 12     # codons (~35 nt)
    rnap_gap: int = 5            # codons between the polymerase and the lead ribosome
    mrna_life: float = 180.0     # s (mean functional lifetime)
    T: float = 2.0e4
    T_burn: float = 600.0
    seed: int = 1
    p_on: float = 0.0            # promoter switching (0 = always on)
    p_off: float = 0.0
    territory_radius: float = 0.0  # um; 0 = no pairing between separate transcripts off the DNA
    bp_len: float = 3.4
    refresh: int = 25
    kernel: Kernel = field(default_factory=Kernel)


def simulate_locus(p: LocusParams) -> dict:
    lib = _lib()
    vterr = 4 / 3 * np.pi * (p.territory_radius * 1e-5) ** 3 * 1e3 if p.territory_radius > 0 else 0.0  # litre
    par = np.array([p.L, p.k_tx, p.v_tx, p.lam, p.v_rib, min(p.kterm, p.v_rib), p.footprint, p.rnap_footprint,
                    p.rnap_gap, p.mrna_life, p.T, p.T_burn, p.seed, p.p_on, p.p_off, vterr, p.bp_len, 1.0,
                    p.refresh], dtype=np.float64)
    kern = p.kernel.array(p.v_rib, p.L)
    out = np.zeros(12)
    lib.simulate_locus(*(a.ctypes.data_as(DP) for a in (par, kern, out)))
    n = out[0:4].sum()
    t = out[4]
    return dict(n=int(n), free=out[0] / n, cis=out[1] / n, sibling=out[2] / n, territory=out[3] / n,
                alpha=out[1:4].sum() / n, protein_rate=n / t, dimer_rate=out[1:4].sum() / 2 / t,
                transcripts=out[5] / t, nascent=out[6] / t, ribosomes=out[7] / t, on_fraction=out[8] / t,
                tx_rate=out[9] / t, from_nascent=out[10] / n)
