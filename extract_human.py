"""Extract per-codon footprint counts (monosome, disome, total translatome; HEK293-T, two
replicates each) for every gene listed in Table S1 of Bertolini et al., plus library totals.
Writes data/disp_human.npz. Run with:  .venv/bin/python -I extract_human.py
"""
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ctd"))
import xlsx  # noqa: E402  (stdlib-only reader, no package import needed)

FILES = {"mono1": "GSM4594340_Mono_HEK_Rep1", "dis1": "GSM4594341_Dis_HEK_Rep1",
         "mono2": "GSM4594342_Mono_HEK_Rep2", "dis2": "GSM4594343_Dis_HEK_Rep2",
         "tot1": "GSM4594348_Total_HEK_Rep1", "tot2": "GSM4594349_Total_HEK_Rep2"}


def main():
    rows = xlsx.read(os.path.join(HERE, "data", "bertolini", "Table_S1.xlsx"))["Co-Co Assembly Candidates"][2:]
    rows = [r + [None] * (5 - len(r)) for r in rows if r and r[0] and r[1] is not None]
    table = {r[0]: (int(r[1]), int(r[2] or 0), float(r[4] or 0.0)) for r in rows}
    hs = {k: h5py.File(os.path.join(HERE, "data", "raw", v + ".h5"), "r") for k, v in FILES.items()}
    genes = [g for g in table if all(g in h for h in hs.values())]
    # control group: genes without any sign of co-co assembly (not in Table S1), long enough for a stop-aligned profile
    ctrl = [g for g in hs["tot1"].keys() if g not in table and all(g in h for h in hs.values())
            and int(hs["tot1"][g].attrs["cds_length"]) >= 1500]
    n_listed = len(genes)
    genes = genes + ctrl
    table.update({g: (0, -1, 0.0) for g in ctrl})
    length = np.array([int(np.ceil(int(hs["tot1"][g].attrs["cds_length"]) / 3)) for g in genes])
    off = np.concatenate([[0], np.cumsum(length)])
    out = dict(genes=np.array(genes), length=length, offset=off,
               onset=np.array([table[g][0] for g in genes]), high=np.array([table[g][1] for g in genes]),
               rpkm=np.array([table[g][2] for g in genes]))
    for k, h in hs.items():
        buf = np.zeros(off[-1], dtype=np.float32)
        for i, g in enumerate(genes):
            a = h[g][...]
            idx = np.minimum((a[0].astype(np.int64) - 1) // 3, length[i] - 1)
            np.add.at(buf, off[i] + idx, a[1])
        out[k] = buf
        out[k + "_total"] = np.array(sum(float(h[g][1].sum()) for g in h.keys()))
        print(k, "reads on listed genes", float(buf.sum()), "library", float(out[k + "_total"]))
    np.savez_compressed(os.path.join(HERE, "data", "disp_human.npz"), **out)
    print("listed genes", n_listed, "| control genes", len(ctrl), "| high confidence", int((out["high"] == 1).sum()), "| codons", int(off[-1]))


if __name__ == "__main__":
    main()
