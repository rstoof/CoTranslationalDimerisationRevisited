"""Three-dimensional dynamics of two nascent chains on a tethered pair of ribosomes.

Wraps chain3d.c. The geometry (shape of both ribosomes, tunnel exits, RNA anchor points) comes
from PDB 9GFT through ctd.pairgeom. Chain motion (microseconds) and translation (seconds) run on
very different time scales. A time-lapse therefore alternates between adding residues and a
stretch of chain and ribosome motion that is far shorter than the translation step it stands for.

What happens before the first pairing is the part of the model that is used for numbers (contact
times, effective concentration). After pairing the model only keeps the two chains together and
gives the paired stretch the geometry of a coiled coil; that part serves the pictures.
"""
from __future__ import annotations

import ctypes
import platform
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np

from . import pairgeom as pg

_HERE = Path(__file__).resolve().parent
_LIB = None
DP = ctypes.POINTER(ctypes.c_double)
FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int)
UP = ctypes.POINTER(ctypes.c_uint64)
MAXB = 160

RES_PER_BEAD = 4
BOND = 3.8 * RES_PER_BEAD            # A
BEAD_RADIUS_H = 7.0                  # A, hydrodynamic radius of one bead
RIBOSOME_RADIUS = 125.0              # A, hydrodynamic radius of a ribosome
KT, ETA = 4.28e-21, 0.69e-3          # J at 37 C; Pa s, water at 37 C
D_BEAD = KT / (6 * np.pi * ETA * BEAD_RADIUS_H * 1e-10) * 1e20                    # A^2/s
D_RIBOSOME = KT / (6 * np.pi * ETA * RIBOSOME_RADIUS * 1e-10) * 1e20              # A^2/s
D_ROT_RIBOSOME = KT / (8 * np.pi * ETA * (RIBOSOME_RADIUS * 1e-10) ** 3)          # rad^2/s
RIGID_EVERY = 2                      # ribosome moves are tried every second sweep
ACCEPT_RATIO = 0.61                  # realised displacement of beads over that of ribosomes, per attempted move
CAPTURE = 12.0                       # A, distance below which two beads are in contact
HELIX_BOND = 1.5 * RES_PER_BEAD      # A, distance between two paired beads along the chain (rise of an alpha-helix)
BEND = 40.0                          # kT, bending stiffness of a paired stretch
PAIRED_STEP = 0.3                    # step of a paired bead relative to that of a free bead
PER_UM = 6.022e-10                   # molecules per A^3 in a 1 uM solution


def _lib():
    global _LIB
    if _LIB is None:
        so, src = _HERE / "libchain3d.so", _HERE / "chain3d.c"
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            arch = ["-arch", platform.machine()] if sys.platform == "darwin" else []
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", *arch, "-o", str(so), str(src), "-lm"], check=True)
        _LIB = ctypes.CDLL(str(so))
        _LIB.run.argtypes = [DP, FP, DP, IP, ctypes.c_double, FP, DP, IP, ctypes.c_double, DP, ctypes.c_int,
                             DP, DP, DP, ctypes.c_int, DP, ctypes.c_int, IP, DP, DP, DP, DP, DP, DP, ctypes.c_long, DP, UP]
        _LIB.run.restype = ctypes.c_int
    return _LIB


_DS = None


def disome():
    global _DS
    if _DS is None:
        _DS = pg.Disome()
    return _DS


class Pair3D:
    """State of the simulation: poses of both ribosomes, both chains, paired interface beads."""

    def __init__(self, extra_codons=40, seed=1, length=395, interface=(11, 148), gate_residues=100, tunnel=30,
                 v=15.0, step=3.0, p_nucleation=1.0, mobility=1.0):
        d = disome()
        self.d, self.rng = d, np.random.default_rng(seed)
        self.seed, self.L, self.v, self.tunnel = seed, length, v, tunnel
        self.spacing = int(d.psite_nt[0] - d.psite_nt[1]) // 3 + extra_codons
        self.i0, self.i1 = (interface[0] - 1) // RES_PER_BEAD, int(np.ceil(interface[1] / RES_PER_BEAD))
        self.gate = int(np.ceil(gate_residues / RES_PER_BEAD))
        self.contour = (d.linker_nt + 3 * extra_codons) * pg.NT_LEN
        (R, t), = pg.tethered_placements(d, 3 * extra_codons, 1, self.rng)[0]
        self.pose1 = np.concatenate([np.eye(3).ravel(), np.zeros(3)])
        self.pose2 = np.concatenate([np.asarray(R, float).ravel(), np.asarray(t, float)])
        self.xa, self.xb = np.zeros((MAXB, 3)), np.zeros((MAXB, 3))
        self.na = self.nb = 0
        self.bound = np.zeros(MAXB, dtype=np.int32)
        self.attached = [1, 1]
        self.rib = [1, 1]                # ribosomes still on the RNA
        self._frozen = None              # display frame once ribosome 1 (and later ribosome 2) has left
        self.sweeps = 0
        self.first_pair = self.first_contact = -1.0
        self.rng_state = np.zeros(4, dtype=np.uint64)
        self.reset_counters()
        # ribosome steps: Stokes-Einstein ratio of the diffusion coefficients, corrected for the acceptance rates
        dtr = step * np.sqrt(BEAD_RADIUS_H / RIBOSOME_RADIUS * ACCEPT_RATIO * mobility * RIGID_EVERY)
        self.par = np.array([BOND, 0.25, step, pg.BEAD, 10.0, CAPTURE, self.i0, self.i1, self.gate, p_nucleation, self.contour,
                             1.5 / (self.contour * pg.B_RNA), dtr, 1.5 * dtr / RIBOSOME_RADIUS, 3.0, 0.004,
                             1, 1, 1, 3, seed * 1_000_003 + 17, 0, 1, 1, RIGID_EVERY, HELIX_BOND, BEND, PAIRED_STEP],
                            dtype=np.float64)
        g1, g2 = d.grid1, d.grid2
        self._g = [np.ascontiguousarray(g.clear, dtype=np.float32) for g in (g1, g2)]
        self._o = [np.ascontiguousarray(g.origin, dtype=np.float64) for g in (g1, g2)]
        self._n = [np.array(g.clear.shape, dtype=np.int32) for g in (g1, g2)]
        self._h = [g1.h, g2.h]
        self._pts2 = np.ascontiguousarray(d.pts2, dtype=np.float64)
        self._fixed = [np.ascontiguousarray(x, dtype=np.float64)
                       for x in (d.exit1, d.exit2, d.q1, d.q2, d.centre[1], d.centre[2])]

    # ------------------------------------------------------------------ geometry
    def _pose(self, k):
        P = self.pose1 if k == 0 else self.pose2
        return P[:9].reshape(3, 3), P[9:]

    def anchor(self, chain):
        R, t = self._pose(chain)
        return R @ (self.d.exit1 if chain == 0 else self.d.exit2) + t

    def direction(self, chain):
        return self._pose(chain)[0] @ (self.d.u1 if chain == 0 else self.d.u2)

    def display(self):
        """Rotation and shift that take simulation coordinates to the frame of ribosome 1. Once
        ribosome 1 has left, the frame moves with ribosome 2; once both have left, it stays."""
        if self.rib[0]:
            R1, t1 = self._pose(0)
            return R1.T.copy(), -R1.T @ t1
        if self.rib[1]:
            Rf, tf = self._frozen
            R2, t2 = self._pose(1)
            Rd = Rf @ R2.T
            return Rd, tf - Rd @ t2
        return self._frozen

    def display_pose2(self):
        Rd, td = self.display()
        R2, t2 = self._pose(1)
        return Rd @ R2, Rd @ t2 + td

    def beads_for(self, codon):
        """Beads outside the tunnel when the ribosome sits at this codon."""
        return max(min(codon, self.L) - self.tunnel, 0) // RES_PER_BEAD

    def add_bead(self, chain):
        x, n = (self.xa, self.na) if chain == 0 else (self.xb, self.nb)
        x[n] = self.anchor(chain) + 2.0 * self.direction(chain) + self.rng.normal(scale=0.5, size=3)
        if chain == 0:
            self.na += 1
        else:
            self.nb += 1

    def set_codons(self, codon_a, codon_b):
        """Grow the chains to the lengths that belong to these ribosome positions."""
        while self.rib[0] and self.na < self.beads_for(codon_a):
            self.add_bead(0)
        while self.rib[1] and self.nb < self.beads_for(codon_b):
            self.add_bead(1)

    def release(self, chain):
        """Termination: the residues still in the tunnel leave it, the chain is set free and its
        ribosome leaves the RNA. A chain that is released as a monomer diffuses away within the time
        of one translation step, and is removed."""
        total = self.L // RES_PER_BEAD
        while (self.na if chain == 0 else self.nb) < total:
            self.add_bead(chain)
        frozen = self.display_pose2() if chain == 0 else self.display()
        self.attached[chain] = 0
        self.rib[chain] = 0
        self._frozen = frozen
        if not self.bound[:self.i1].any():
            if chain == 0:
                self.na = 0
            else:
                self.nb = 0

    # ------------------------------------------------------------------ dynamics
    def reset_counters(self):
        self.count = dict(sweeps=0, any=0.0, nterm=0.0, pairs=0.0, bead_msd=0.0, bead_sweeps=0, rib_msd=0.0, rib_ang=0.0,
                          rib_sweeps=0, accept=0.0, accept_rigid=0.0)
        self.first_contact = -1.0
        self.count_from = self.sweeps

    def run(self, nsweeps):
        lib, n = _lib(), int(nsweeps)
        par = self.par.copy()
        par[16], par[17] = self.attached
        par[18] = par[22] = self.rib[0]
        par[23] = self.rib[1]
        par[21] = self.sweeps
        stats = np.zeros(11)
        stats[2], stats[8] = self.first_pair, self.first_contact
        p = lambda a, T: a.ctypes.data_as(T)
        lib.run(p(par, DP), p(self._g[0], FP), p(self._o[0], DP), p(self._n[0], IP), self._h[0],
                p(self._g[1], FP), p(self._o[1], DP), p(self._n[1], IP), self._h[1],
                p(self._pts2, DP), len(self._pts2), p(self.pose1, DP), p(self.pose2, DP),
                p(self.xa, DP), self.na, p(self.xb, DP), self.nb, p(self.bound, IP),
                *(p(a, DP) for a in self._fixed), n, p(stats, DP), p(self.rng_state, UP))
        self.sweeps += n
        self.first_pair, self.first_contact = stats[2], stats[8]
        c = self.count
        c["sweeps"] += n; c["any"] += stats[5]; c["nterm"] += stats[6]; c["pairs"] += stats[7]
        c["accept"] += stats[0] * n; c["accept_rigid"] += stats[1] * n
        if self.na + self.nb > 0:
            c["bead_msd"] += stats[4] * n; c["bead_sweeps"] += n
        if self.rib[0] + self.rib[1] > 0:
            c["rib_msd"] += stats[9] * n; c["rib_ang"] += stats[10] * n; c["rib_sweeps"] += n
        return dict(accept=stats[0], accept_rigid=stats[1], paired=int(stats[3]))

    @property
    def sweep_time(self):
        """Physical time of one sweep (s): the beads diffuse as spheres of radius BEAD_RADIUS_H in water."""
        c = self.count
        return c["bead_msd"] / c["bead_sweeps"] / (6 * D_BEAD) if c["bead_sweeps"] else np.nan

    def summary(self):
        """Counters since the last reset, with the realised diffusion coefficients of the ribosomes."""
        c, dt = self.count, self.sweep_time
        hi = min(self.na, self.nb, self.i1)
        vol = 4 / 3 * np.pi * CAPTURE ** 3 * PER_UM
        n, rs = max(c["sweeps"], 1), max(c["rib_sweeps"], 1)
        return dict(sweeps=c["sweeps"], sweep_time=dt, n_pairs=max(hi - self.i0, 0),
                    first_contact=self.first_contact - self.count_from if self.first_contact >= 0 else -1.0,
                    contact_sweeps=c["any"], ceff_nterm=c["nterm"] / n / vol,
                    ceff_interface=c["pairs"] / n / max(hi - self.i0, 1) / vol,
                    d_ribosome=c["rib_msd"] / rs / (6 * dt), d_rot=c["rib_ang"] / rs / (6 * dt),
                    accept=c["accept"] / n, accept_rigid=c["accept_rigid"] / n, exposed=(self.na * RES_PER_BEAD, self.nb * RES_PER_BEAD))

    def frame(self, **info):
        """State in the display frame (ribosome 1 at its coordinates in the structure)."""
        Rd, td = self.display()
        R2, t2 = self.display_pose2()
        c1 = self.d.centre[1]
        c2 = R2 @ self.d.centre[2] + t2
        e1, e2 = self.d.exit1, R2 @ self.d.exit2 + t2
        return dict(info, sweeps=self.sweeps, pose=np.concatenate([R2.ravel(), t2]).astype(np.float32),
                    xa=(self.xa[:self.na] @ Rd.T + td).astype(np.float32), xb=(self.xb[:self.nb] @ Rd.T + td).astype(np.float32),
                    bound=self.bound[:self.i1].astype(np.int8), attached=tuple(self.attached), rib=tuple(self.rib),
                    exit_distance=float(np.linalg.norm(e2 - e1)), centre_distance=float(np.linalg.norm(c2 - c1)))


def timelapse(seed=1, extra_codons=40, sweeps_competent=1_750_000, sweeps_other=100_000, fine=5_000, clips=False,
              clip_frames=24, event_before=24, event_after=72, after_release=3, max_steps=None, **kw):
    """Translate both chains from initiation until both are released.

    Both ribosomes advance one bead (4 codons, 0.27 s at 15 codons per second) at a time, the leading
    one `spacing` codons ahead. Every step is followed by `sweeps_competent` sweeps of motion while both
    chains are competent and not yet paired, and by `sweeps_other` sweeps otherwise: before that nothing
    can pair, afterwards nothing remains to be decided. Returns (steps, meta, movie): one summary per
    step; with clips=True, movie holds the frames (one every `fine` sweeps) of the end of every step and
    of the stretch around the first pairing. Keeping frames does not change the trajectory."""
    s = Pair3D(extra_codons=extra_codons, seed=seed, **kw)
    steps, clip_list, event, recent = [], [], None, deque(maxlen=event_before)
    gaps, pair_gap = [], None           # distance between the tunnel exits while both chains wait for each other
    step = extra = 0
    while max_steps is None or step < max_steps:
        pos_b = step * RES_PER_BEAD
        if s.rib[0] and pos_b + s.spacing >= s.L:
            s.release(0)                                       # the leading ribosome reaches the stop codon
        if s.rib[1] and pos_b >= s.L:
            s.release(1)                                       # the trailing ribosome reaches the stop codon
        info = dict(step=step, time=pos_b / s.v, codon_a=min(pos_b + s.spacing, s.L), codon_b=min(pos_b, s.L))
        s.set_codons(info["codon_a"], info["codon_b"])
        waiting = s.rib == [1, 1] and s.first_pair < 0 and min(s.na, s.nb, s.i1) - s.i0 >= s.gate
        n_fine = (sweeps_competent if waiting else sweeps_other) // fine
        clip, j = deque(maxlen=clip_frames), 0
        while j < n_fine:
            before = s.first_pair
            s.run(fine); j += 1
            if s.rib == [1, 1]:
                gap = float(np.linalg.norm(s.anchor(1) - s.anchor(0)))
                if before < 0 <= s.first_pair:
                    pair_gap = gap
                    n_fine = j + max(sweeps_other // fine, event_after)      # paired: a short stretch more, then on
                elif waiting and s.first_pair < 0:
                    gaps.append(gap)
            if clips:
                fr = s.frame(**info)
                if event is None and before < 0 <= s.first_pair:
                    event = dict(frames=list(recent) + [fr], left=event_after, step=step, index=len(recent))
                elif event is not None and event["left"] > 0:
                    event["frames"].append(fr); event["left"] -= 1
                recent.append(fr); clip.append(fr)
        fr = s.frame(**info)
        steps.append(dict(info, paired=int(fr["bound"].sum()), exit_distance=fr["exit_distance"], sweeps=j * fine, waiting=bool(waiting),
                          centre_distance=fr["centre_distance"], rib=fr["rib"], na=s.na, nb=s.nb))
        if clips:
            clip_list.append(list(clip))
        if not s.rib[1]:
            extra += 1
            if extra > after_release:
                break
        step += 1
    first = next((st for st in steps if st["paired"] > 0), None)
    meta = dict(seed=seed, extra_codons=extra_codons, spacing=s.spacing, L=s.L, i0=s.i0, i1=s.i1, gate=s.gate, v=s.v,
                tunnel=s.tunnel, contour=s.contour, res_per_bead=RES_PER_BEAD, sweeps_competent=sweeps_competent,
                sweeps_other=sweeps_other, fine=fine, sweep_time=s.sweep_time, first_pair_sweep=s.first_pair,
                first_pair_step=first["step"] if first else None, first_pair_codon=first["codon_b"] if first else None,
                waiting_steps=sum(st["waiting"] for st in steps), waiting_sweeps=sum(st["sweeps"] for st in steps if st["waiting"]),
                exit_gap_first_pair=pair_gap,
                exit_gap_waiting=(float(np.mean(gaps)), *np.percentile(gaps, [10, 50, 90]).tolist()) if gaps else None)
    return steps, meta, (dict(clips=clip_list, event=event) if clips else None)


def contact_statistics(seed, extra_codons=40, codon_b=200, sweeps=12_000_000, chunk=100_000, mobility=1.0,
                       stop_at_contact=False, **kw):
    """Contacts between the interfaces of two competent chains of fixed length, without pairing.
    Gives the time to the first contact from an equilibrated start (the pairing time if every contact
    led to a dimer) and the fraction of time spent in contact (an effective concentration)."""
    s = Pair3D(extra_codons=extra_codons, seed=seed, p_nucleation=0.0, mobility=mobility, **kw)
    for c in range(0, codon_b + 1, RES_PER_BEAD):
        s.set_codons(c + s.spacing, c); s.run(1000)
    s.run(200_000)
    s.reset_counters()
    while s.count["sweeps"] < sweeps and not (stop_at_contact and s.first_contact >= 0):
        s.run(chunk)
    return dict(s.summary(), seed=seed, mobility=mobility, extra_codons=extra_codons, codon_b=codon_b, spacing=s.spacing)
