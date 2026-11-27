"""Structure-based geometry of two neighbouring E. coli ribosomes (PDB 9GFT, collided disome):
tunnel exits, shortest free path between them, nascent-chain Monte Carlo with the real
excluded volume, and an ensemble of randomly oriented, mRNA-tethered ribosome pairs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .structure import Grid, chain_table, flood_to_bulk, kabsch, read_cif

DATA = Path(__file__).resolve().parent.parent / "data"
BEAD = 5.0          # A; minimum clearance for a chain bead
NT_LEN = 5.9        # A per nucleotide of single-stranded RNA
B_RNA = 20.0        # A; Kuhn length of single-stranded mRNA


def kuhn(n):
    """Kuhn segments (count, length) reproducing Rg = 1.93 n^0.598 A for an unfolded chain."""
    rg2 = (1.93 * n**0.598) ** 2
    b = 6 * rg2 / (n * 3.8)
    return max(int(round(n * 3.8 / b)), 1), b


class Disome:
    def __init__(self, path=DATA / "pdb" / "9GFT.cif.gz"):
        tb, at = read_cif(path)
        ch = chain_table(tb, at)
        xyz = at["xyz"].astype(np.float64)
        sel = lambda a: at["asym"] == a
        desc = {c["asym"]: c["desc"] for c in ch}
        by_desc = lambda d: [c["asym"] for c in ch if c["desc"] == d]
        s23, s16 = by_desc("23S RNA"), by_desc("16S ribosomal RNA")
        vemp = sel(by_desc("VemP nascent chain")[0])
        ca = vemp & (at["atom"] == "CA")
        vca = xyz[ca][np.argsort(at["seq"][ca])]
        # ribosome 1 = stalled (carries the nascent chain), ribosome 2 = collided (trailing)
        d23 = [np.linalg.norm(xyz[sel(a)].mean(0) - vca.mean(0)) for a in s23]
        a23_1, a23_2 = (s23[0], s23[1]) if d23[0] < d23[1] else (s23[1], s23[0])
        c23 = {1: xyz[sel(a23_1)].mean(0), 2: xyz[sel(a23_2)].mean(0)}
        a16 = {}
        for a in s16:
            c = xyz[sel(a)].mean(0)
            a16[1 if np.linalg.norm(c - c23[1]) < np.linalg.norm(c - c23[2]) else 2] = a
        cen = {k: xyz[sel(a16[k]) | sel(a23_1 if k == 1 else a23_2)].mean(0) for k in (1, 2)}
        rib = {}
        for c in ch:
            if "HrpA" in c["desc"]:
                rib[c["asym"]] = -1
            elif c["desc"] == "messenger RNA":
                rib[c["asym"]] = 0
            else:
                rib[c["asym"]] = 1 if np.linalg.norm(c["centroid"] - cen[1]) < np.linalg.norm(c["centroid"] - cen[2]) else 2
        r = np.array([rib[a] for a in at["asym"]])
        self.xyz, self.rib, self.at = xyz, r, at
        self.centre = cen
        body = (r >= 0) & ~vemp
        self.grid = Grid(xyz[body], 3.0, 36.0)                       # both ribosomes + mRNA
        self.grid1 = Grid(xyz[(r == 1) & ~vemp], 3.0, 36.0)          # each ribosome on its own
        self.grid2 = Grid(xyz[r == 2], 3.0, 36.0)
        # tunnel exit of ribosome 1 by flooding from the N-terminus of the traced nascent chain
        self.ptc1, self.start1 = vca[-1], vca[0]
        self.exit1, self.tunnel_geodesic, _ = flood_to_bulk(self.grid, self.start1)
        # map onto ribosome 2 by superposing the 23S rRNA backbones
        P1, P2 = sel(a23_1) & (at["atom"] == "P"), sel(a23_2) & (at["atom"] == "P")
        s1, s2 = dict(zip(at["seq"][P1], xyz[P1])), dict(zip(at["seq"][P2], xyz[P2]))
        common = sorted(set(s1) & set(s2))
        A, B = np.array([s1[k] for k in common]), np.array([s2[k] for k in common])
        self.R12, self.t12 = kabsch(A, B)
        self.rmsd23 = float(np.sqrt((((A @ self.R12.T + self.t12) - B) ** 2).sum(1).mean()))
        self.exit2 = self.R12 @ self.exit1 + self.t12
        self.start2 = self.R12 @ self.start1 + self.t12
        self.u1 = (self.exit1 - self.start1) / np.linalg.norm(self.exit1 - self.start1)
        self.u2 = self.R12 @ self.u1
        # mRNA backbone (5'->3'): 5' part in ribosome 2, 3' part in ribosome 1
        m = (r == 0) & (at["atom"] == "P")
        o = np.argsort(at["seq"][m])
        self.mrna, self.mrna_seq = xyz[m][o], at["seq"][m][o]
        d1 = np.linalg.norm(self.mrna - xyz[sel(a16[1])].mean(0), axis=1)
        d2 = np.linalg.norm(self.mrna - xyz[sel(a16[2])].mean(0), axis=1)
        self.psite_nt = (int(self.mrna_seq[np.argmin(d1)]), int(self.mrna_seq[np.argmin(d2)]))
        # points where the linker leaves ribosome 2 (3' side) and enters ribosome 1 (5' side)
        link = np.where((d1 > 40) & (d2 > 40) & (self.mrna_seq > self.psite_nt[1]) & (self.mrna_seq < self.psite_nt[0]))[0]
        self.q2, self.q1 = self.mrna[link[0]], self.mrna[link[-1]]
        self.linker_nt = int(link[-1] - link[0])
        # coarse point sets for clash tests
        back = np.isin(at["atom"], ["P", "CA"])
        self.pts1 = xyz[(r == 1) & back][::3]
        self.pts2 = xyz[(r == 2) & back][::3]

    # ------------------------------------------------------------------ summary numbers
    def summary(self):
        e = self.exit2 - self.exit1
        d = np.linalg.norm(e)
        line = self.exit1[None] + np.linspace(0, 1, 201)[:, None] * e[None]
        return dict(
            centre_distance=float(np.linalg.norm(self.centre[1] - self.centre[2])),
            exit_distance=float(d),
            tunnel_length=float(np.linalg.norm(self.exit1 - self.ptc1)),
            rmsd_23S=self.rmsd23,
            angle_between_tunnels=float(np.degrees(np.arccos(self.u1 @ self.u2))),
            angle_exit1_to_exit2=float(np.degrees(np.arccos(self.u1 @ e / d))),
            angle_exit2_to_exit1=float(np.degrees(np.arccos(self.u2 @ -e / d))),
            line_blocked_fraction=float((self.grid.clearance(line) < 4).mean()),
            psite_spacing_nt=self.psite_nt[0] - self.psite_nt[1],
            linker_nt=self.linker_nt, linker_span=float(np.linalg.norm(self.q1 - self.q2)),
        )

    def free_path(self, spacing=6.0, probe=BEAD):
        """Shortest path between the two exits through free space (Dijkstra on a 26-neighbour lattice)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        g = self.grid
        lo = np.minimum(self.exit1, self.exit2) - 260
        hi = np.maximum(self.exit1, self.exit2) + 260
        ax = [np.arange(l, h, spacing) for l, h in zip(lo, hi)]
        X = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
        free = g.clearance(X.reshape(-1, 3)).reshape(X.shape[:3]) >= probe
        idx = -np.ones(free.shape, dtype=np.int64)
        idx[free] = np.arange(free.sum())
        rows, cols, w = [], [], []
        sh = free.shape
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dy, dz) <= (0, 0, 0):
                        continue
                    a = idx[max(0, -dx):sh[0] - max(0, dx), max(0, -dy):sh[1] - max(0, dy), max(0, -dz):sh[2] - max(0, dz)]
                    b = idx[max(0, dx):sh[0] - max(0, -dx), max(0, dy):sh[1] - max(0, -dy), max(0, dz):sh[2] - max(0, -dz)]
                    ok = (a >= 0) & (b >= 0)
                    rows.append(a[ok]); cols.append(b[ok])
                    w.append(np.full(ok.sum(), spacing * np.sqrt(dx * dx + dy * dy + dz * dz)))
        n = int(free.sum())
        G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        node = lambda p: idx[tuple(np.round((p - lo) / spacing).astype(int))]
        a, b = node(self.exit1), node(self.exit2)
        return float(dijkstra(G, directed=False, indices=a)[b])


# ---------------------------------------------------------------------- nascent-chain Monte Carlo
def grow_chains(clear_fn, tether, n_res, n_keep, rng, batch=120_000, max_batches=400):
    """N-terminal positions of freely jointed chains that start at `tether` and stay in free space.
    Chains are grown segment by segment and discarded at the first collision (unbiased)."""
    nk, b = kuhn(n_res)
    ends, tried, kept = [], 0, 0
    for _ in range(max_batches):
        pos = np.repeat(tether[None], batch, 0)
        for _ in range(nk):
            v = rng.normal(size=(len(pos), 3))
            v *= b / np.linalg.norm(v, axis=1, keepdims=True)
            new = pos + v
            ok = (clear_fn(new) >= BEAD) & (clear_fn(pos + 0.5 * v) >= BEAD)
            pos = new[ok]
            if not len(pos):
                break
        tried += batch
        if len(pos):
            ends.append(pos); kept += len(pos)
        if kept >= n_keep:
            break
    ends = np.concatenate(ends)[:n_keep] if ends else np.zeros((0, 3))
    return ends, kept / max(tried, 1)


def overlap(e1, w1, e2, w2, bin_w=16.0):
    """Effective molar concentration from the overlap of two end-point clouds.
    w1, w2: statistical weight per point (1/number of sampled conformations)."""
    if not len(e1) or not len(e2):
        return 0.0
    lo = np.maximum(e1.min(0), e2.min(0)) - bin_w
    hi = np.minimum(e1.max(0), e2.max(0)) + bin_w
    if np.any(hi <= lo):
        return 0.0
    edges = [np.arange(l, h + bin_w, bin_w) for l, h in zip(lo, hi)]
    h1, _ = np.histogramdd(e1, edges)
    h2, _ = np.histogramdd(e2, edges)
    return float((h1 * h2).sum() * w1 * w2 / bin_w**3 / 6.02214e-4)


def random_rotations(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a, b, c, d = q.T
    return np.stack([np.stack([a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], -1),
                     np.stack([2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)], -1),
                     np.stack([2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d], -1)], 1)


def tethered_placements(ds: Disome, extra_nt, n, rng, max_clash=0.004, max_trials=400_000):
    """Rigid-body placements (R, t) of ribosome 2 with random orientation, tethered to
    ribosome 1 by an mRNA linker of the collided length plus `extra_nt` nucleotides.
    Returns (placements, acceptance): orientations that would make the two ribosomes overlap are rejected."""
    contour = (ds.linker_nt + extra_nt) * NT_LEN
    sig = np.sqrt(contour * B_RNA / 3)
    out, tried = [], 0
    while len(out) < n and tried < max_trials:
        m = 2000
        Rs = random_rotations(m, rng)
        r = rng.normal(scale=sig, size=(m, 3))
        keep = np.linalg.norm(r, axis=1) <= contour
        for R, rv in zip(Rs[keep], r[keep]):
            tried += 1
            # ribosome 2 rotates about its linker anchor q2, which is then moved to q1 - rv
            t = (ds.q1 - rv) - R @ ds.q2
            # cheap pre-test on the centres, then the full backbone test
            if np.linalg.norm(R @ ds.centre[2] + t - ds.centre[1]) < 150.0:
                continue
            p = ds.pts2 @ R.T + t
            if (ds.grid1.clearance(p) < 3.0).mean() <= max_clash:
                out.append((R, t))
                if len(out) >= n:
                    break
    return out, len(out) / max(tried, 1)
