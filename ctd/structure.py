"""Minimal mmCIF reader and geometric analysis of ribosome pairs (no external dependencies)."""
from __future__ import annotations

import gzip
import re

import numpy as np

_TOK = re.compile(r"""'((?:[^']|'(?=\S))*)'(?=\s|$)|"((?:[^"]|"(?=\S))*)"(?=\s|$)|(\S+)""")


def _tokens(line):
    return [a or b or c for a, b, c in _TOK.findall(line)]


def read_cif(path, want=("_entity.", "_struct_asym.", "_entity_poly.")):
    """Return (tables, atoms). tables[category] = list of dict rows; atoms = dict of numpy arrays."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        lines = f.read().split("\n")
    tables, i, n = {}, 0, len(lines)
    atoms = None
    while i < n:
        ln = lines[i]
        if ln.startswith("loop_"):
            i += 1
            cols = []
            while i < n and lines[i].startswith("_"):
                cols.append(lines[i].strip()); i += 1
            cat = cols[0].split(".")[0] + "."
            if cat == "_atom_site.":
                j = i
                while j < n and not (lines[j].startswith("#") or lines[j].startswith("loop_") or lines[j].startswith("_")):
                    j += 1
                atoms = _atoms(cols, lines[i:j]); i = j
            elif cat in want:
                names = [c.split(".")[1] for c in cols]
                rows, vals = [], []
                while i < n and not (lines[i].startswith("#") or lines[i].startswith("loop_") or lines[i].startswith("_")):
                    if lines[i].startswith(";"):
                        txt = [lines[i][1:]]; i += 1
                        while not lines[i].startswith(";"):
                            txt.append(lines[i]); i += 1
                        vals.append(" ".join(txt).strip())
                    else:
                        vals.extend(_tokens(lines[i]))
                    i += 1
                    while len(vals) >= len(names):
                        rows.append(dict(zip(names, vals[:len(names)]))); vals = vals[len(names):]
                tables[cat] = rows
            continue
        i += 1
    return tables, atoms


def _atoms(cols, rows):
    names = [c.split(".")[1] for c in cols]
    ix = {k: names.index(k) for k in ("type_symbol", "label_atom_id", "label_comp_id", "label_asym_id",
                                       "label_entity_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
                                       "auth_asym_id", "auth_seq_id")}
    sp = [r.split() for r in rows if r and not r.startswith("#")]
    get = lambda k: [s[ix[k]] for s in sp]
    return dict(
        xyz=np.array([[s[ix["Cartn_x"]], s[ix["Cartn_y"]], s[ix["Cartn_z"]]] for s in sp], dtype=np.float32),
        element=np.array(get("type_symbol")), atom=np.array([a.strip('"') for a in get("label_atom_id")]),
        comp=np.array(get("label_comp_id")), asym=np.array(get("label_asym_id")),
        entity=np.array(get("label_entity_id")), auth_asym=np.array(get("auth_asym_id")),
        seq=np.array([int(x) if x.lstrip("-").isdigit() else -9999 for x in get("auth_seq_id")]),
    )


def chain_table(tables, atoms):
    """One row per chain: (asym, auth_asym, entity, description, n_atoms, centroid)."""
    desc = {r["id"]: r.get("pdbx_description", "?") for r in tables.get("_entity.", [])}
    out = []
    for a in dict.fromkeys(atoms["asym"]):
        m = atoms["asym"] == a
        e = atoms["entity"][m][0]
        out.append(dict(asym=a, auth=atoms["auth_asym"][m][0], entity=e, desc=desc.get(e, "?"),
                        n=int(m.sum()), centroid=atoms["xyz"][m].mean(0)))
    return out


# ----------------------------------------------------------------------------- geometry helpers
class Grid:
    """Occupancy grid with a clearance field (distance in A to the nearest atom-containing voxel)."""

    def __init__(self, xyz, spacing=3.0, margin=36.0):
        from scipy import ndimage
        self.h = float(spacing)
        self.origin = xyz.min(0) - margin
        shape = np.ceil((xyz.max(0) + margin - self.origin) / self.h).astype(int) + 1
        occ = np.zeros(shape, dtype=bool)
        ijk = np.floor((xyz - self.origin) / self.h).astype(int)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        self.occ = occ
        self.clear = (ndimage.distance_transform_edt(~occ) * self.h).astype(np.float32)

    def index(self, p):
        return np.floor((np.asarray(p) - self.origin) / self.h).astype(int)

    def centre(self, ijk):
        return self.origin + (np.asarray(ijk) + 0.5) * self.h

    def clearance(self, pts, outside=99.0):
        """Clearance at arbitrary points; points outside the grid are in open solvent."""
        ijk = np.floor((np.asarray(pts) - self.origin) / self.h).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.occ.shape)), axis=-1)
        out = np.full(ijk.shape[:-1], outside, dtype=np.float32)
        g = ijk[ok]
        out[ok] = self.clear[g[:, 0], g[:, 1], g[:, 2]]
        return out


def kabsch(P, Q):
    """Rotation R and translation t with R @ p + t ~ q (least squares)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def flood_to_bulk(grid, start, probe=4.5, bulk=13.0, max_steps=80):
    """Breadth-first flood through free space from `start` until open solvent is reached.
    Returns (exit_point, geodesic_length_A, visited_mask)."""
    from scipy import ndimage
    free = grid.clear >= probe
    s = tuple(grid.index(start))
    if not free[s]:  # snap to the nearest free voxel
        cand = np.argwhere(free)
        s = tuple(cand[np.argmin(((cand - np.array(s)) ** 2).sum(1))])
    front = np.zeros_like(free); front[s] = True
    seen = front.copy()
    st = ndimage.generate_binary_structure(3, 1)
    for step in range(1, max_steps + 1):
        new = ndimage.binary_dilation(front, st) & free & ~seen
        if not new.any():
            return None, step * grid.h, seen
        seen |= new
        hit = new & (grid.clear >= bulk)
        if hit.any():
            idx = np.argwhere(hit)
            best = idx[np.argmax(grid.clear[hit])]
            return grid.centre(best), step * grid.h, seen
        front = new
    return None, max_steps * grid.h, seen
