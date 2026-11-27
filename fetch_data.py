"""Download the third-party data used in this repository.

    python3 fetch_data.py ecoli      # 14 files, ~100 MB: DiSP profiles of Bertolini et al. 2021 in E. coli (GEO GSE151959)
    python3 fetch_data.py human      #  6 files, ~280 MB: HEK293-T monosome, disome and total translatome profiles
    python3 fetch_data.py tables     # Table S1 of Bertolini et al. 2021 (list of co-co assembly candidates), from Europe PMC
    python3 fetch_data.py pdb        # PDB entries 9GFT, 4AC0, 1LBI, 3BDN (already included in data/pdb/)
    python3 fetch_data.py all        # ecoli + human + tables
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"

ECOLI = {
    4594368: "Mono_Ecoli_LMNC_Rep1", 4594369: "Dis_Ecoli_LMNC_Rep1",
    4594370: "Mono_Ecoli_LMNC_Rep2", 4594371: "Dis_Ecoli_LMNC_Rep2",
    4594372: "Mono_Ecoli_1B-Cherry_Rep1", 4594373: "Dis_Ecoli_1B-Cherry_Rep1",
    4594374: "Mono_Ecoli_1B-Cherry_Rep2", 4594375: "Dis_Ecoli_1B-Cherry_Rep2",
    4594376: "Mono_Ecoli_1B_-Cherry_Rep1", 4594377: "Dis_Ecoli_1B_-Cherry_Rep1",
    4594378: "Mono_Ecoli_1B_-Cherry_Rep2", 4594379: "Dis_Ecoli_1B_-Cherry_Rep2",
    4594380: "Mono_Ecoli_DCTN1_Rep1", 4594381: "Dis_Ecoli_DCTN1_Rep1",
}
HUMAN = {
    4594340: "Mono_HEK_Rep1", 4594341: "Dis_HEK_Rep1",
    4594342: "Mono_HEK_Rep2", 4594343: "Dis_HEK_Rep2",
    4594348: "Total_HEK_Rep1", 4594349: "Total_HEK_Rep2",
}


def fetch(samples):
    RAW.mkdir(parents=True, exist_ok=True)
    for gsm, name in samples.items():
        fn = f"GSM{gsm}_{name}.h5"
        dest = RAW / fn
        if dest.exists() and dest.stat().st_size > 0:
            print("have ", fn)
            continue
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{str(gsm)[:4]}nnn/GSM{gsm}/suppl/{fn}"
        req = urllib.request.Request(url, headers={"User-Agent": "generic-research-client/1.0"})
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        print(f"fetched {fn}  {dest.stat().st_size / 1e6:.1f} MB")


def _get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print("have ", dest.name)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "generic-research-client/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"fetched {dest.name}  {dest.stat().st_size / 1e6:.2f} MB")


def fetch_tables():
    """Table S1 is supplementary material of the Science paper; it is not redistributed here."""
    _get("https://europepmc.org/api/fulltextRepo?pprId=PMC7613021&type=FILE&fileName=EMS146497-supplement-Table_S1.xlsx"
         "&mimeType=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         RAW.parent / "bertolini" / "Table_S1.xlsx")


def fetch_pdb():
    for name in ("9GFT.cif.gz", "4AC0-assembly1.cif.gz", "1LBI-assembly1.cif.gz", "3BDN-assembly1.cif.gz"):
        _get("https://files.rcsb.org/download/" + name, RAW.parent / "pdb" / name)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ecoli"
    if which in ("ecoli", "human", "all"):
        fetch({"ecoli": ECOLI, "human": HUMAN, "all": {**ECOLI, **HUMAN}}[which])
    if which in ("tables", "all"):
        fetch_tables()
    if which == "pdb":
        fetch_pdb()
