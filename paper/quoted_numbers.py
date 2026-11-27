"""Every number quoted in paper/manuscript.md, computed from the result files.

    python3 paper/quoted_numbers.py        # prints the list
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ctd import circuits, window  # noqa: E402

V, TAU_R = 15.0, 10 / 15.0


def numbers():
    R = pickle.load(open(ROOT / "results" / "tests.pkl", "rb"))
    G = pickle.load(open(ROOT / "results" / "geometry.pkl", "rb"))
    K = pickle.load(open(ROOT / "results" / "kernel.pkl", "rb"))
    M = pickle.load(open(ROOT / "results" / "model.pkl", "rb"))
    N = {}
    E, H, S = R["ecoli"], R["human"], G["summary"]
    for rep in ("1", "2"):
        a = E[("1B", rep)]
        N[f"f{rep}"], N[f"f{rep}_se"], N[f"mid{rep}"] = a["f_plateau"], a["f_plateau_se"], a["fit"]["mid"]
        N[f"exp{rep}"] = a["fit"]["mid"] - 30 - 10                      # coil 1B starts at codon 11
        N[f"mut{rep}"] = E[("1Bmut", rep)]["E_plateau"]
        N[f"E{rep}"] = a["E_plateau"]
        b = E[("LMNC", rep)]
        N[f"lam_f{rep}"], N[f"lam_mid{rep}"], N[f"lam_res{rep}"] = b["f_plateau"], b["fit"]["mid"], b["fit"]["mid"] + 22
    N["dctn_mid"] = E[("DCTN1", "1")]["fit"]["mid"]
    d = np.load(ROOT / "data" / "disp_ecoli.npz")
    sh = [100 * float(d[f"1B_{f}_{r}"].sum()) / (float(d[f"1B_{f}_{r}"].sum()) + float(d[f"1B_{f}_{r}_host"])) for r in "12" for f in ("mono", "dis")]
    N["share_lo"], N["share_hi"] = min(sh), max(sh)
    N["reads_min"] = min(float(d[f"1B_{f}_{r}"].sum()) for r in "12" for f in ("mono", "dis"))
    f = R["fit"]
    x = f["x"]; sel = (x > 20) & (x < 340)
    N["rep_noise"] = float(np.sqrt(np.mean(((E[("1B", "1")]["f"][:39] - E[("1B", "2")]["f"][:39]) / 2)[sel] ** 2)))
    for k in ("paper", "gated_paper", "constant", "structure"):
        N[f"rmse_{k}"], N[f"spacing_{k}"], N[f"alpha_{k}"], N[f"a_{k}"], N[f"gate_{k}"], N[f"lam_{k}"] = (
            f[k]["rmse"], f[k]["spacing"], f[k]["alpha"], f[k]["a"], f[k]["n_on"], f[k]["lam"])
    N["end_model"], N["end_data"] = float(f["structure"]["f"][-1]), float(f["target"][-1])
    # geometry
    N.update(centre_nm=S["centre_distance"] / 10, exit_nm=S["exit_distance"] / 10, ang1=S["angle_exit1_to_exit2"],
             ang2=S["angle_exit2_to_exit1"], blocked=100 * S["line_blocked_fraction"], path_nm=G["free_path"] / 10,
             nmin=G["free_path"] / 2 / 3.6, tunnel_A=S["tunnel_length"], rmsd=S["rmsd_23S"], psite_nt=S["psite_spacing_nt"],
             s_min=int(G["s_min"]), paper_nm=int(G["s_min"]) * 0.9)
    dm = dict(zip(G["n_trail"].tolist(), G["direct"].tolist()))
    for n in (100, 120, 150, 200, 300):
        N[f"col{n}"] = dm[n]
    ng = list(K["n_grid"])
    for e in K["extras"]:
        r = K["ensemble"][int(e)]
        N[f"acc{int(e)}"], N[f"face{int(e)}"], N[f"top{int(e)}"] = 100 * r["accept"], 100 * r["facing"], 100 * r["top1_share"]
        N[f"c100_{int(e)}"] = r["table"][ng.index(100), ng.index(100)]
    N["collided_scale"] = K["collided_scale"]
    v = K["validation"]
    N["shortcut_max_factor"] = float(max(10 ** abs(np.log10(v[e]["gauss"][:, j].mean() / v[e]["hist"][:, j].mean()))
                                         for e in v for j in range(len(v[e]["pairs"]))))
    cf = M["ceff_fit"]
    N.update(c_inf=cf["c_inf"], s0=cf["s0"], m=cf["m"])
    for key in ("point", "footprint", "structure"):
        N[f"val_{key}_rms"], N[f"val_{key}_max"] = M[key]["stats"]["rms"], M[key]["stats"]["max"]
    N["val_const_long_rms"] = M["structure"]["stats_const_long"]["rms"]
    N["val_n"] = int(M["point"]["alpha"].size + M["footprint"]["alpha"].size + M["structure"]["alpha"].size)
    # human
    on = H["onset"]
    N.update(n_high=H["n_high"], on_med=float(np.median(on)), on_q1=float(np.percentile(on, 25)), on_q3=float(np.percentile(on, 75)),
             on_p10=float(np.percentile(on, 10)), on_min_codon=30 + G["free_path"] / 2 / 3.6,
             on_before_min=100 * float(np.mean(on < 30 + G["free_path"] / 2 / 3.6)),
             on_frac=float(np.median((on - 30) / H["length"])))
    om = H["onset_meta"]
    pl = om["enrich"][(om["x"] > 100) & (om["x"] < 300)].mean()
    N["meta_plateau"] = float(pl)
    N["meta_half"] = float(om["x"][np.where(om["enrich"] > 1 + 0.5 * (pl - 1))[0][0]])
    N["meta_90"] = float(om["x"][np.where(om["enrich"] > 1 + 0.9 * (pl - 1))[0][0]])
    sm = H["stop_meta"]
    N.update(stop_end=float(sm["high"]["enrich"][-1]), stop_lo=float(sm["high"]["ci"][0][-1]), stop_hi=float(sm["high"]["ci"][1][-1]),
             stop_ctrl=float(sm["control"]["enrich"][-1]), stop_n=sm["high"]["n"], ctrl_n=sm["control"]["n"])
    xs = sm["high"]["x"]; N["stop_start"] = float(-xs[np.where(sm["high"]["ci"][1] >= 1.0)[0][-1] + 1])
    e = H["eff"]
    N.update(eff_n=len(e["eff"]), eff_med=float(np.median(e["eff"])), rho_remaining=float(e["rho_remaining"][0]),
             p_remaining=float(e["rho_remaining"][1]), rho_rpkm=float(e["rho_rpkm"][0]))
    # comparison with equation 3 of the 2020 paper
    Lm = R["lambda"]
    for lam, tag in ((0.0126, "a"), (0.05, "b"), (0.126, "c")):
        i = int(np.argmin(abs(Lm["lams"] - lam)))
        N[f"cmp_lam_{tag}"], N[f"cmp_new_{tag}"], N[f"cmp_old_{tag}"] = float(Lm["lams"][i]), float(Lm["gate 100, kon 1e6"]["alpha"][i]), float(Lm["paper"][i])
    # locus
    Lc = R["locus"]; kt = Lc["ktx"]; lab = "N-terminal interface"
    pick = lambda terr, q: Lc[(lab, terr)][int(np.argmin(abs(kt - q)))]
    rr = Lc[(lab, "territory 0.2 um")]
    N["cis_lo"], N["cis_hi"] = min(r["cis"] for r in rr), max(r["cis"] for r in rr)
    for q, tag in ((0.01, "001"), (0.1, "01"), (1.0, "1")):
        N[f"sib_{tag}"] = 100 * float(np.mean([pick(t, q)["sibling"] for t in ("DNA-tethered only", "territory 0.2 um", "whole cell")]))
        N[f"terr_{tag}"] = 100 * pick("territory 0.2 um", q)["territory"]
        N[f"cell_{tag}"] = 100 * pick("whole cell", q)["territory"]
    N["nascent_01"], N["transcripts_01"] = pick("territory 0.2 um", 0.1)["nascent"], pick("territory 0.2 um", 0.1)["transcripts"]
    N["from_nascent"] = 100 * max(r["from_nascent"] for t in ("DNA-tethered only", "territory 0.2 um", "whole cell") for r in Lc[(lab, t)])
    tx = np.array([q["tx_rate"] for q in rr]); prot = np.array([q["protein_rate"] for q in rr])
    cis = np.array([q["cis"] for q in rr]) * prot / 2
    tr = np.array([(q["sibling"] + q["territory"]) if q["sibling"] * q["n"] >= 6 else np.nan for q in rr]) * prot / 2
    ok = np.isfinite(tr)
    N["slope_cis"], N["slope_trans"] = float(np.polyfit(np.log(tx), np.log(cis), 1)[0]), float(np.polyfit(np.log(tx[ok]), np.log(tr[ok]), 1)[0])
    b = Lc["bursty"]["res"]
    btx = np.array([q["tx_rate"] for q in b]); btr = np.array([(q["sibling"] + q["territory"]) * q["protein_rate"] / 2 for q in b])
    N["slope_switch"] = float(np.polyfit(np.log(btx), np.log(btr), 1)[0])
    N["sib_native"], N["sib_fast"] = 100 * Lc["speed"][15.0]["sibling"], 100 * Lc["speed"][75.0]["sibling"]
    # repressors
    P = R["repressors"]
    for nm, tag in (("TetR(B)", "tet"), ("LacI", "lac"), ("lambda cI", "ci")):
        p = P[nm]
        N[f"{tag}_n"], N[f"{tag}_q50"], N[f"{tag}_q90"], N[f"{tag}_w50"], N[f"{tag}_w90"] = len(p["residues"]), p["q50"], p["q90"], p["window50"], p["window90"]
        N[f"{tag}_a6"], N[f"{tag}_a7"] = p["alpha"][(0.1, 1e6)], p["alpha"][(0.1, 1e7)]
    win = 270 - P["scan_n_on"]
    a6 = P[("scan", 1e6)][0]
    N["alpha_nterm"] = float(a6[list(win).index(220)])
    N["alpha_c"] = circuits.rep_hopf()
    for k in (0.1, 1.0, np.inf):
        wc = window.critical_window(0.1, k, N["alpha_c"], TAU_R)
        N[f"wc_k{k}"] = wc * V
    N["hill_0"], N["hill_c"], N["hill_01"] = circuits.effective_hill(0.0), circuits.effective_hill(N["alpha_c"]), circuits.effective_hill(0.1)
    N["lamW_c"] = float(-np.log(1 - N["alpha_c"] / (2 - N["alpha_c"])))      # Lambda at which alpha = alpha_c
    return N


if __name__ == "__main__":
    N = numbers()
    for k in sorted(N):
        v = N[k]
        print(f"{k:22s} {v:.4g}" if isinstance(v, (float, np.floating)) else f"{k:22s} {v}")
