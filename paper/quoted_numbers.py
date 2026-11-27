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


def numbers_3d():
    """Numbers of section 2.4 and Methods 4.4 (three-dimensional simulation), from results/sim3d.pkl
    and paper/figures/movie.json."""
    import json
    from ctd import chain3d as c3
    from ctd.sim2 import ceff
    S = pickle.load(open(ROOT / "results" / "sim3d.pkl", "rb")); st = S["stats"]
    N = {}
    for m, r in sorted(st["rate"].items()):
        N[f"3d_rate_x{m:g}"] = (f"{r['rate']:.0f} per s (95% interval {r['lo']:.0f} to {r['hi']:.0f}); {r['events']} of {r['runs']} runs; "
                                f"realised mobility {r['mobility']:.3g}; exposure {r['exposure'] * 1e3:.2f} ms")
    p = st["pooled"]
    N["3d_rate_pooled"], N["3d_rate_pooled_lo"], N["3d_rate_pooled_hi"] = p["rate"], p["lo"], p["hi"]
    N["3d_rate_pooled_events"], N["3d_rate_pooled_runs"], N["3d_rate_pooled_exposure_ms"] = p["events"], p["runs"], p["exposure"] * 1e3
    N["3d_first_contact_ms"] = 1e3 / p["rate"]
    N["3d_power_gamma"], N["3d_power_lo"], N["3d_power_hi"] = (st["power"][k] for k in ("gamma", "lo", "hi"))
    mobs = sorted(st["rate"])
    N["3d_mobility_min"], N["3d_mobility_max"] = st["rate"][mobs[0]]["mobility"], st["rate"][mobs[-1]]["mobility"]
    d_cell = 0.04e8                                              # A^2/s, polysomes in E. coli (Bakshi et al.)
    N["3d_water_over_cell"] = c3.D_RIBOSOME / d_cell
    r0 = st["rate"][mobs[0]]
    N["3d_bound_cell"] = r0["lo"] * d_cell / c3.D_RIBOSOME / r0["mobility"]
    N["3d_bound_cell_point"] = r0["rate"] * d_cell / c3.D_RIBOSOME / r0["mobility"]
    N["3d_turn_us_slowest"] = 1e6 / (6 * c3.D_ROT_RIBOSOME * r0["mobility"])   # time to turn by one radian (rms)
    c = st["ceff_all"]
    N["3d_ceff_nterm_uM"], N["3d_ceff_nterm_lo"], N["3d_ceff_nterm_hi"] = (c["nterm"][k] for k in ("mean", "lo", "hi"))
    N["3d_ceff_interface_uM"], N["3d_ceff_interface_lo"], N["3d_ceff_interface_hi"] = (c["interface"][k] for k in ("mean", "lo", "hi"))
    N["3d_ceff_runs"], N["3d_ceff_mobilities"] = c["nterm"]["runs"], [round(m, 4) for m in sorted(st["ceff"])]
    n_lead, n_trail = st["exposed"]
    N["3d_exposed_lead"], N["3d_exposed_trail"], N["3d_spacing"] = n_lead, n_trail, st["spacing"]
    N["3d_ceff_static_uM"] = float(ceff(n_lead, n_trail, st["spacing"] - 8)) * 1e6
    N["3d_sweep_ps"] = st["sweep_time"] * 1e12
    base = S["contacts"][1.0]
    N["3d_run_us"] = base[0]["sweeps"] * st["sweep_time"] * 1e6
    N["3d_D_ribosome_um2_s"] = float(np.mean([r["d_ribosome"] for r in base])) * 1e-8
    N["3d_D_rot_rad2_s"] = float(np.mean([r["d_rot"] for r in base]))
    N["3d_D_ribosome_stokes"], N["3d_D_rot_stokes"], N["3d_D_bead_um2_s"] = c3.D_RIBOSOME * 1e-8, c3.D_ROT_RIBOSOME, c3.D_BEAD * 1e-8
    N["3d_accept_beads"], N["3d_accept_ribosomes"] = float(np.mean([r["accept"] for r in base])), float(np.mean([r["accept_rigid"] for r in base]))
    lap = S["timelapse"]
    m0 = lap[0][1]
    b, step_time = m0["res_per_bead"], m0["res_per_bead"] / m0["v"]
    t_sweep = float(np.mean([m["sweep_time"] for _, m in lap]))
    N["3d_motion_competent_us"] = m0["sweeps_competent"] * t_sweep * 1e6
    N["3d_motion_other_us"] = m0["sweeps_other"] * t_sweep * 1e6
    N["3d_time_compression"] = step_time / (m0["sweeps_competent"] * t_sweep)
    c_comp = -(-(m0["tunnel"] + (m0["i0"] + m0["gate"]) * b) // b) * b
    c_rel = -(-(m0["L"] - m0["spacing"]) // b) * b
    N["3d_codon_competent"], N["3d_codon_release"] = c_comp, c_rel
    N["3d_time_competent_s"], N["3d_time_release_s"] = c_comp / m0["v"], c_rel / m0["v"]
    # first contacts in the time-lapses: events over the time during which both chains waited for each other
    ev, expo = 0, 0
    for steps_, m in lap:
        first_wait = next((x for x in steps_ if x["waiting"]), None)
        if first_wait is None:
            continue
        if m["first_pair_sweep"] >= 0:
            ev += 1
            expo += m["first_pair_sweep"] - sum(x["sweeps"] for x in steps_ if x["step"] < first_wait["step"])
        else:
            expo += sum(x["sweeps"] for x in steps_ if x["waiting"])
    from scipy.stats import chi2
    ci = np.array([chi2.ppf(0.025, 2 * ev) / 2 if ev else 0.0, chi2.ppf(0.975, 2 * ev + 2) / 2])
    N["3d_lapse_rate"], (N["3d_lapse_rate_lo"], N["3d_lapse_rate_hi"]) = ev / (expo * t_sweep), ci / (expo * t_sweep)
    t_transl = expo / m0["sweeps_competent"] * step_time
    N["3d_lapse_waiting_translation_s"], N["3d_lapse_waiting_motion_ms"] = t_transl, expo * t_sweep * 1e3
    N["3d_lapse_k_translation"], (N["3d_lapse_k_translation_lo"], N["3d_lapse_k_translation_hi"]) = ev / t_transl, ci / t_transl
    N["3d_lapse_paired"], N["3d_lapse_runs"] = st["lapse_paired"], st["lapse_runs"]
    N["3d_lapse_first_codons"] = sorted(c for c in st["lapse_first_codons"] if c is not None)
    N["3d_rna_nt"], N["3d_rna_contour_A"] = round(m0["contour"] / 5.9), m0["contour"]
    b2 = c3.BOND ** 2 + 3 / 0.25                                  # mean-square bond length of a harmonic bond (stiffness 0.25 kT/A^2)
    for res in (100, 220):
        n = res // c3.RES_PER_BEAD - 1
        N[f"3d_rg_ratio_{res}"] = np.sqrt(b2 * n * (n + 2) / (6 * (n + 1))) / (1.93 * res ** 0.598)
    if st.get("gap_first_pair"):
        g = np.array(st["gap_first_pair"]) / 10
        N["3d_gap_first_pair_nm"] = f"mean {g.mean():.1f}, range {g.min():.1f} to {g.max():.1f} (n = {len(g)})"
        gw = np.array(st["gap_waiting"]) / 10
        N["3d_gap_waiting_nm"] = f"mean {gw[0]:.1f}, 10th to 90th percentile {gw[1]:.1f} to {gw[3]:.1f}"
    try:
        info = json.loads((ROOT / "paper" / "figures" / "movie.json").read_text())
        N["3d_movie_seed"], N["3d_movie_first_pair_codon"], N["3d_movie_frames"] = info["seed"], info["first_pair_codon"], info["frames"]
        N["3d_movie_frame_ns"], N["3d_movie_zip_us"] = info["frame_ns"], info.get("zip_us")
        N["3d_movie_clip_us"] = info["frame_ns"] * info.get("clip_frames", 20) / 1e3
        m = next(mm for _, mm in lap if mm["seed"] == info["seed"])
        if m.get("exit_gap_first_pair") is not None:
            N["3d_movie_gap_first_pair_nm"] = m["exit_gap_first_pair"] / 10
            N["3d_movie_gap_waiting_nm"] = m["exit_gap_waiting"][0] / 10
        for name, sh in info["shots"].items():
            N[f"3d_shot_{name}"] = f"{sh['time']:.1f} s, codons {sh['codon_a']}/{sh['codon_b']}, {sh['paired']} residues paired"
    except FileNotFoundError:
        pass
    return N


if __name__ == "__main__":
    N = numbers()
    if (ROOT / "results" / "sim3d.pkl").exists():
        N.update(numbers_3d())
    for k in sorted(N):
        v = N[k]
        print(f"{k:22s} {v:.4g}" if isinstance(v, (float, np.floating)) else f"{k:22s} {v}")
