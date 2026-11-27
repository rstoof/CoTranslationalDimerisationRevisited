"""Figures of the manuscript (paper/manuscript.md). Run from the repository root:

    python3 paper/make_figures.py

Needs results/tests.pkl, geometry.pkl, kernel.pkl, model.pkl and data/pdb/9GFT.cif.gz.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ctd import circuits, disp, window  # noqa: E402
from ctd.sim2 import ceff  # noqa: E402

FIG = ROOT / "paper" / "figures"
R = pickle.load(open(ROOT / "results" / "tests.pkl", "rb"))
G = pickle.load(open(ROOT / "results" / "geometry.pkl", "rb"))
K = pickle.load(open(ROOT / "results" / "kernel.pkl", "rb"))
M = pickle.load(open(ROOT / "results" / "model.pkl", "rb"))

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.8, "lines.markersize": 5,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "axes.axisbelow": True,
})
ONE_PER_CELL = 1.66e-9
V = 15.0                 # elongation rate (residues per second)
TAU_R = 10 / V           # minimal headway of two ribosomes: footprint of 10 codons
S_MIN = 8                # codons between two collided ribosomes


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom", color=INK)


def feature_bar(ax, feats, y=1.03, h=0.05):
    tr = ax.get_xaxis_transform()
    for name, a, b in feats:
        ax.add_patch(Rectangle((a, y), b - a, h, transform=tr, clip_on=False, fc=GRID, ec=AXIS, lw=0.8))
        ax.text((a + b) / 2, y + h / 2, name, transform=tr, ha="center", va="center", fontsize=7.5, color=INK2)


# ------------------------------------------------------------------------------- shared panels
def panel_structure(ax):
    """Projection of the two collided ribosomes (PDB 9GFT) onto the plane of the two tunnel exits."""
    from ctd import pairgeom as pg
    d = pg.Disome()
    e1 = d.exit2 - d.exit1; dist = np.linalg.norm(e1); e1 /= dist
    um = d.u1 + d.u2; e2 = um - (um @ e1) * e1; e2 /= np.linalg.norm(e2)
    o = (d.exit1 + d.exit2) / 2
    pr = lambda p: np.stack([(p - o) @ e1, (p - o) @ e2], -1) / 10.0
    back = np.isin(d.at["atom"], ["P", "CA"])
    for k, col in ((1, C[0]), (2, C[1])):
        p = pr(d.xyz[(d.rib == k) & back][::4]); ax.scatter(p[:, 0], p[:, 1], s=1.2, color=col, alpha=0.35, linewidths=0)
    m = pr(d.mrna); ax.plot(m[:, 0], m[:, 1], color=INK, lw=1.6)
    for ex, u in ((d.exit1, d.u1), (d.exit2, d.u2)):
        p = pr(ex); q = pr(ex + 70 * u)
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
        ax.plot(*p, "o", color="white", mec=INK, mew=1.5, ms=7, zorder=5)
    p1, p2 = pr(d.exit1), pr(d.exit2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=INK, lw=1, ls=(0, (3, 3)))
    ax.text(0, p1[1] + 1.6, f"{dist / 10:.0f} nm", ha="center", fontsize=9, bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(0, -14.5, "RNA", ha="center", fontsize=8)
    ax.text(p1[0] - 1.5, p1[1] + 7.5, "exit tunnel", ha="right", fontsize=8, color=INK2)
    ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("nm"); ax.set_ylabel("nm"); ax.set_xlim(-28, 28); ax.set_ylim(-34, 11)
    ax.legend(handles=[Patch(color=C[0], label="leading ribosome"), Patch(color=C[1], label="trailing ribosome")],
              fontsize=8, loc="lower center", ncol=2, handlelength=1)


def panel_window(ax):
    """Competence windows of three consecutive nascent chains."""
    v, L, nc = 10.0, 300, 130                       # illustration: 300 residues, interface exposed after 130 codons
    starts = [0.0, 7.0, 31.0]
    for i, t0 in enumerate(starts):
        y = 2 - i
        ax.add_patch(Rectangle((t0, y - 0.22), nc / v, 0.44, fc=GRID, ec=AXIS, lw=0.8))
        ax.add_patch(Rectangle((t0 + nc / v, y - 0.22), (L - nc) / v, 0.44, fc=C[0] if i < 2 else C[3], ec="none", alpha=0.9))
        ax.text(t0 - 0.8, y, f"chain {i + 1}", ha="right", va="center", fontsize=8.5)
    t_a, t_b = starts[1] + nc / v, starts[0] + L / v
    ax.add_patch(Rectangle((t_a, 0.72), t_b - t_a, 1.56, fc=C[1], ec="none", alpha=0.18))
    ax.annotate("", (starts[1], 2.48), (starts[0], 2.48), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    ax.text((starts[0] + starts[1]) / 2, 2.6, r"$\Delta t$", ha="center", fontsize=9)
    ax.annotate("", (starts[0] + L / v, 2.48), (starts[0] + nc / v, 2.48), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    ax.text(starts[0] + (nc + L) / 2 / v, 2.6, r"$W$", ha="center", fontsize=9)
    ax.text((t_a + t_b) / 2, 1.5, "both competent\n" + r"$W-\Delta t$", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(starts[2] + (nc + L) / 2 / v, -0.55, "no partner in time:\nreleased as monomer", ha="center", va="top", fontsize=8, color=INK2)
    ax.text(starts[0] + nc / 2 / v, 2, "interface\nnot exposed", ha="center", va="center", fontsize=7.2, color=INK2)
    ax.set_xlim(-9, 64); ax.set_ylim(-1.3, 3.0); ax.set_yticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False); ax.set_xlabel("time since initiation of chain 1 (s)")
    ax.legend(handles=[Patch(color=GRID, label="translating, not competent"), Patch(color=C[0], label="competent: interface exposed")],
              fontsize=8, loc="lower left", bbox_to_anchor=(0.0, -0.02))


# ---------------------------------------------------------------------------------------- figure 1
def fig1():
    E = R["ecoli"]; f = R["fit"]; x = f["x"]
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.7), constrained_layout=True, width_ratios=[1, 1.35, 0.62])
    ax = axs[0]
    for rep, ls in (("1", "-"), ("2", (0, (4, 2)))):
        r = E[("1B", rep)]; ax.plot(r["x"], r["E"], color=C[0], ls=ls, label=f"coiled coil, replicate {rep}")
        r = E[("1Bmut", rep)]; ax.plot(r["x"], r["E"], color=C[1], ls=ls, label=f"1B* mutant, replicate {rep}")
    ax.set_yscale("log"); ax.set_ylim(0.4, 60); ax.set_xlim(0, 395)
    ax.legend(fontsize=7.2, loc="upper left"); feature_bar(ax, disp.CONSTRUCTS["1B"]["features"])
    ax.set_xlabel("codon"); ax.set_ylabel("disome / monosome footprints\n(relative to the first 120 codons)")
    panel(ax, "A")
    ax = axs[1]
    lo = np.minimum(E[("1B", "1")]["f"][:39], E[("1B", "2")]["f"][:39]); hi = np.maximum(E[("1B", "1")]["f"][:39], E[("1B", "2")]["f"][:39])
    ax.vlines(x, lo, hi, color=MUTED, lw=1.2)
    ax.plot(x, f["target"], "o", color=INK, ms=4, label="measured (mean of two replicates)", zorder=5)
    names = {"paper": "overlapping spheres [1]", "gated_paper": "overlapping spheres + interface gate",
             "constant": "competence window, constant rate", "structure": "competence window, structure-based rate"}
    for i, k in enumerate(("paper", "gated_paper", "constant", "structure")):
        ax.plot(x, f[k]["f"], color=C[i], lw=2, ls=(0, (4, 2)) if k == "paper" else "-", label=names[k])
    ax.axvspan(340, 395, color=GRID, alpha=0.6, lw=0); ax.text(367, 0.03, "not\nfitted", ha="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 395); ax.set_ylim(-0.03, 1.0); ax.set_xlabel("codon"); ax.set_ylabel("fraction of ribosomes in a\nnascent-chain-linked pair")
    ax.legend(fontsize=7.4, loc="upper left"); feature_bar(ax, disp.CONSTRUCTS["1B"]["features"])
    panel(ax, "B")
    ax = axs[2]
    ks = ("paper", "gated_paper", "constant", "structure")
    v = [f[k]["rmse"] for k in ks]
    ax.barh(range(4), v, color=C[:4], height=0.55)
    for i, val in enumerate(v):
        ax.text(val + 0.004, i, f"{val:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(range(4), ["spheres", "spheres\n+ gate", "window,\nconstant", "window,\nstructure"], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, max(v) * 1.3); ax.grid(axis="y", visible=False); ax.set_xlabel("RMSE")
    panel(ax, "C")
    save(fig, "fig1_disp")


# ---------------------------------------------------------------------------------------- figure 2
def fig2():
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4), constrained_layout=True)
    panel_structure(axs[0, 0]); panel(axs[0, 0], "A")
    n = K["n_grid"]
    ax = axs[0, 1]
    ok = G["direct"] > 0
    ax.plot(G["n_trail"][ok], G["direct"][ok] * 1e6, "o-", color=INK, ms=4, label="collided pair (structure)")
    for i, e in enumerate((5, 20, 80, 320)):
        ax.plot(n, np.diag(K["ensemble"][int(e)]["table"]) * 1e6, color=C[i], label=f"+{e} codons, free orientation")
    ax.axhline(ONE_PER_CELL * 1e6, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(545, ONE_PER_CELL * 1e6 * 1.5, "one molecule per cell", ha="right", fontsize=7.5, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(1e-5, 30); ax.set_xlim(0, 550)
    ax.set_xlabel("exposed residues per chain"); ax.set_ylabel(r"effective concentration $c_{\mathrm{eff}}$ ($\mu$M)")
    ax.legend(fontsize=7.2, loc="lower right"); panel(ax, "B")
    ax = axs[1, 0]
    ex = np.concatenate([[0], K["extras"]]) + G["s_min"]
    for i, nn in enumerate((60, 100, 200)):
        j = list(n).index(nn)
        ax.plot(ex, K["table"][j, j, :] * 1e6, "o-", color=C[i], ms=4, mec="white", label=f"{nn} residues")
    cf = M["ceff_fit"]; sf = np.logspace(1, np.log10(330), 100)
    ax.plot(sf, cf["c_inf"] / (1 + (cf["s0"] / sf) ** cf["m"]) * 1e6, color=INK, lw=1.2, ls=(0, (3, 2)), label="equation 1")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-5, 30)
    ax.set_xlabel("distance between ribosomes (codons)"); ax.set_ylabel(r"$c_{\mathrm{eff}}$ ($\mu$M)")
    ax.legend(fontsize=8, loc="lower right", title="exposed", title_fontsize=8); panel(ax, "C")
    ax = axs[1, 1]
    e = K["extras"]
    acc = np.array([K["ensemble"][int(q)]["accept"] for q in e]) * 100
    face = np.array([K["ensemble"][int(q)]["facing"] for q in e]) * 100
    ax.plot(e + G["s_min"], acc, "o-", color=C[0], ms=4, mec="white", label="orientations without steric clash")
    ax.plot(e + G["s_min"], np.maximum(face, 1e-3), "s-", color=C[1], ms=4, mec="white", label="of these: both exits facing each other")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-3, 150)
    ax.set_xlabel("distance between ribosomes (codons)"); ax.set_ylabel("random orientations (%)")
    ax.legend(fontsize=7.5, loc="lower right"); panel(ax, "D")
    save(fig, "fig2_geometry")


# ---------------------------------------------------------------------------------------- figure 3
def fig3():
    fig, axs = plt.subplots(2, 2, figsize=(9.8, 7.6), constrained_layout=True)
    panel_window(axs[0, 0]); panel(axs[0, 0], "A")
    lam_f = np.logspace(-2.6, np.log10(0.6), 200)
    ax = axs[0, 1]
    iw = list(M["windows"]).index(270)
    Wsec = (270 + 1) / M["v"]
    for i, k in enumerate(M["ks"]):
        ax.plot(lam_f, window.alpha(lam_f, Wsec, k, TAU_R), color=C[i], label=f"$k$ = {k:g} s$^{{-1}}$")
        ax.plot(M["footprint"]["lam_eff"][i, iw], M["footprint"]["alpha"][i, iw], "o", color=C[i], ms=4.5, mec="white", mew=0.7)
    ax.plot(lam_f, window.alpha(lam_f, Wsec, np.inf, TAU_R), color=INK, lw=1.2, ls=(0, (2, 2)), label=r"$k\to\infty$")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"initiation rate $\lambda$ (s$^{-1}$)"); ax.set_ylabel(r"co-translational fraction $\alpha$")
    ax.legend(fontsize=7.6, loc="upper left"); ax.set_title("competence window of 270 codons (18 s)", fontweight="normal", fontsize=9)
    panel(ax, "B")
    ax = axs[1, 0]
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    for key, mk, col, lab in (("point", "o", C[0], "constant rate, point ribosomes"), ("footprint", "s", C[1], "constant rate, footprint of 10 codons"),
                              ("structure", "^", C[2], "structure-based rate")):
        ax.plot(M[key]["pred"].ravel(), M[key]["alpha"].ravel(), mk, color=col, ms=3.2, alpha=0.75, mec="none", label=lab)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha$, equations 2 to 4"); ax.set_ylabel(r"$\alpha$, stochastic simulation")
    ax.legend(fontsize=7.2, loc="upper left"); panel(ax, "C")
    ax = axs[1, 1]
    lam = np.logspace(-3, np.log10(0.6), 121); win = np.linspace(0, 400, 101)
    A = window.alpha(lam[None, :], win[:, None] / V, 1.0, TAU_R)
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    mesh = ax.pcolormesh(lam, win, A, cmap=cm, vmin=0, vmax=1, shading="gouraud")
    cs = ax.contour(lam, win, A, levels=[0.022, 0.1, 0.5, 0.9], colors=[INK], linewidths=0.8)
    ax.clabel(cs, fmt="%g", fontsize=7.5)
    ax.axvline(0.1, color=C[1], lw=1.2, ls=(0, (4, 3))); ax.text(0.108, 150, r"measured $\lambda$ [17]", color=C[1], fontsize=8, va="center", ha="left", rotation=90)
    ax.set_xscale("log"); ax.grid(False)
    ax.set_xlabel(r"initiation rate $\lambda$ (s$^{-1}$)"); ax.set_ylabel("residues translated after the interface\nhas left the tunnel")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02); cb.outline.set_visible(False); cb.set_label(r"$\alpha$ ($k$ = 1 s$^{-1}$)")
    panel(ax, "D")
    save(fig, "fig3_model")


# ---------------------------------------------------------------------------------------- figure 4
def fig4():
    H = R["human"]
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.5), constrained_layout=True)
    ax = axs[0]
    m = H["onset_meta"]
    ax.plot(m["x"], m["enrich"], color=C[0], lw=2)
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (2, 2))); ax.axhline(1, color=AXIS, lw=1)
    ax.set_xlabel("codons from the assembly onset"); ax.set_ylabel("disome / monosome footprints (fold)")
    panel(ax, "A")
    ax = axs[1]
    s_ = H["stop_meta"]
    ax.fill_between(s_["high"]["x"], s_["high"]["ci"][0], s_["high"]["ci"][1], color=C[0], alpha=0.22, lw=0)
    ax.plot(s_["high"]["x"], s_["high"]["enrich"], color=C[0], lw=2, label=f"co-co assembly candidates (n = {s_['high']['n']})")
    ax.plot(s_["control"]["x"], s_["control"]["enrich"], color=C[1], lw=2, label=f"other proteins (n = {s_['control']['n']})")
    ax.axhline(1, color=AXIS, lw=1); ax.set_ylim(0.5, 1.2)
    ax.set_xlabel("codons before the stop codon"); ax.set_ylabel("disome / monosome footprints\n(relative to $-$450 to $-$300)")
    ax.legend(fontsize=7.6, loc="lower left"); panel(ax, "B")
    ax = axs[2]
    e = H["eff"]
    ax.scatter(e["remaining"], e["eff"], s=6, color=C[0], alpha=0.35, linewidths=0)
    q = np.quantile(e["remaining"], np.linspace(0, 1, 7))
    xm = [np.median(e["remaining"][(e["remaining"] >= a) & (e["remaining"] <= b)]) for a, b in zip(q[:-1], q[1:])]
    ym = [np.median(e["eff"][(e["remaining"] >= a) & (e["remaining"] <= b)]) for a, b in zip(q[:-1], q[1:])]
    ax.plot(xm, ym, "o-", color=INK, ms=5, label="median per sextile")
    ax.set_xscale("log"); ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("codons translated after the onset"); ax.set_ylabel("assembly efficiency [2]")
    ax.text(0.97, 0.05, rf"Spearman $\rho$ = {e['rho_remaining'][0]:.2f}", transform=ax.transAxes, ha="right", fontsize=8.5)
    ax.legend(fontsize=8, loc="upper left"); panel(ax, "C")
    save(fig, "fig4_human")


# ---------------------------------------------------------------------------------------- figure 5
def fig5():
    P = R["repressors"]; H = R["human"]; Lc = R["locus"]
    fig, axs = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    ax = axs[0, 0]
    for i, nm in enumerate(["TetR(B)", "LacI", "lambda cI"]):
        p = P[nm]; L = p["L"]
        ax.add_patch(Rectangle((0, i - 0.28), 1, 0.56, fc="white", ec=AXIS, lw=1))
        ax.add_patch(Rectangle((1 - 30 / L, i - 0.28), 30 / L, 0.56, fc=GRID, ec="none"))
        ax.vlines(p["residues"] / L, i - 0.26, i + 0.26, color=C[0], lw=1.1)
        ax.plot([p["q50"] / L], [i - 0.4], "v", color=INK, ms=6)
    fr = (H["onset"] - 30) / H["length"]
    q = np.percentile(fr, [10, 25, 50, 75, 90]); i = 3
    ax.add_patch(Rectangle((q[1], i - 0.2), q[3] - q[1], 0.4, fc=C[1], ec="none", alpha=0.75))
    ax.plot([q[0], q[4]], [i, i], color=C[1], lw=1.5); ax.plot([q[2], q[2]], [i - 0.2, i + 0.2], color=INK, lw=1.8)
    ax.set_yticks(range(4), ["TetR (207 AA)", "LacI (360 AA)", "λ cI (237 AA)", "co-co assembling\nproteins (829, human)"], fontsize=8.5)
    ax.set_ylim(3.6, -0.75); ax.set_xlim(0, 1); ax.grid(False)
    ax.set_xlabel("position in the protein (fraction of its length)")
    ax.text(0.99, -0.55, "grey: still inside the\ntunnel at release", ha="right", va="center", fontsize=7.5, color=INK2)
    panel(ax, "A")
    ax = axs[0, 1]
    win = 300 - 30 - P["scan_n_on"]
    wf = np.linspace(0, 272, 300)
    def rate(win_codons):
        n_on = 300 - 30 - win_codons
        return lambda sep, t_ov: 1e6 * ceff(n_on + t_ov * V / 2 + sep, n_on + t_ov * V / 2, max(sep - S_MIN, 0.0))
    wf = np.linspace(2, 272, 91)
    for (lam, col) in ((0.1, C[0]), (1.0, C[1])):
        j = list(P["scan_lams"]).index(lam)
        lam_eff = lam if lam < 0.3 else 0.585        # realised initiation rate with a footprint of 10 codons
        a = [window.alpha_kernel(lam_eff, (x + 1) / V, rate(x), V, TAU_R) for x in wf]
        ax.plot(wf, np.maximum(a, 1e-5), color=col, label=rf"$\lambda$ = {lam:g} s$^{{-1}}$, equations 2 to 4")
        ax.plot(win, np.maximum(P[("scan", 1e6)][j], 1e-5), "o", color=col, ms=4.5, mec="white", mew=0.7, label="stochastic simulation")
    ax.axhline(0.022, color=INK, lw=1, ls=(0, (2, 2))); ax.text(268, 0.026, r"repressilator stops oscillating ($\alpha_c$ = 0.022)", fontsize=7.5, ha="right")
    for nm, short in (("TetR(B)", "TetR"), ("LacI", "LacI"), ("lambda cI", "cI")):
        wv = P[nm]["window50"]
        ax.axvline(wv, color=MUTED, lw=1, ls=(0, (1, 2))); ax.text(wv + 3, 1.6e-4, f"{short}\n({wv})", fontsize=7.5, va="bottom")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5); ax.set_xlim(0, 272)
    ax.set_xlabel("residues translated after the interface has left the tunnel"); ax.set_ylabel(r"co-translational fraction $\alpha$")
    ax.legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.03)); panel(ax, "B")
    ax = axs[1, 0]
    wf = np.linspace(0, 272, 200)
    for (lam, col) in ((0.01, C[2]), (0.1, C[0]), (0.5, C[1])):
        a = window.alpha(lam, wf / V, 1.0, TAU_R)
        ax.plot(wf, [circuits.effective_hill(float(x)) for x in a], color=col, label=rf"$\lambda$ = {lam:g} s$^{{-1}}$")
    ax.set_ylim(1, 2.02); ax.set_xlim(0, 272)
    ax.set_xlabel("residues translated after the interface has left the tunnel"); ax.set_ylabel("effective Hill exponent")
    ax.legend(fontsize=8, loc="upper right"); panel(ax, "C")
    ax = axs[1, 1]
    lab = "N-terminal interface"
    r = Lc[(lab, "territory 0.2 um")]
    tx = np.array([q_["tx_rate"] for q_ in r]); prot = np.array([q_["protein_rate"] for q_ in r])
    cis = np.array([q_["cis"] for q_ in r]) * prot / 2
    tr = np.array([(q_["sibling"] + q_["territory"]) if q_["sibling"] * q_["n"] >= 6 else np.nan for q_ in r]) * prot / 2
    ok = np.isfinite(tr)
    sl = lambda y, m: np.polyfit(np.log(tx[m]), np.log(y[m]), 1)[0]
    ax.plot(tx, cis, "o-", color=C[0], ms=4, mec="white", label=f"within one polysome (slope {sl(cis, tx > 0):.2f})")
    ax.plot(tx[ok], tr[ok], "o-", color=C[1], ms=4, mec="white", label=f"between transcripts (slope {sl(tr, ok):.2f})")
    b = Lc["bursty"]["res"]
    btx = np.array([q_["tx_rate"] for q_ in b]); btr = np.array([(q_["sibling"] + q_["territory"]) * q_["protein_rate"] / 2 for q_ in b])
    ax.plot(btx, btr, "s--", color=C[2], ms=5, mec="white", lw=1.5, label=f"between transcripts, promoter switching\non and off (slope {np.polyfit(np.log(btx), np.log(btr), 1)[0]:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(top=80)
    ax.set_xlabel(r"transcripts per second"); ax.set_ylabel("co-translational dimers per second")
    ax.legend(fontsize=7.5, loc="upper left"); panel(ax, "D")
    save(fig, "fig5_circuits")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5"]
    for name in which:
        globals()[name]()
    print("figures written to", FIG)
