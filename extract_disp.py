"""Extract per-codon monosome/disome footprint counts for the constructs that
Bertolini et al. over-expressed in E. coli, from the GEO HDF5 files in data/raw/.
Writes data/disp_ecoli.npz. Run with:  .venv/bin/python -I extract_disp.py
"""
import glob
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GENE = {"LMNC": "SFLMNCTS", "1B-Cherry": "SF_Coil1B_WT_mCherry_TS",
        "1B_-Cherry": "SF_Coil1B_Mut_mCherry_TS", "DCTN1": "SF_DCTN1_TS"}
LABEL = {"LMNC": "LMNC", "1B-Cherry": "1B", "1B_-Cherry": "1Bmut", "DCTN1": "DCTN1"}


def codon_counts(dset):
    arr = dset[...]
    n = int(np.ceil(int(dset.attrs["cds_length"]) / 3))
    out = np.zeros(n)
    np.add.at(out, (arr[0].astype(int) - 1) // 3, arr[1])
    return out


def main():
    out = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "data", "raw", "GSM*_Ecoli_*.h5"))):
        base = os.path.basename(fn)[:-3]
        _, frac, _, rest = base.split("_", 3)          # GSM, Mono|Dis, Ecoli, <construct>_RepN
        construct, rep = rest.rsplit("_Rep", 1)
        with h5py.File(fn, "r") as f:
            g = GENE[construct]
            key = f"{LABEL[construct]}_{frac.lower()}_{rep}"
            out[key] = codon_counts(f[g])
            # reads on all endogenous E. coli genes: the reference for library size
            out[key + "_host"] = np.array(sum(float(f[k][1].sum()) for k in f.keys() if k != g))
    np.savez_compressed(os.path.join(HERE, "data", "disp_ecoli.npz"), **out)
    for k in sorted(out):
        if not k.endswith("_host"):
            print(f"{k:16s} codons {len(out[k]):5d} reads {out[k].sum():10.0f} host reads {float(out[k + '_host']):10.0f}")


if __name__ == "__main__":
    main()
