"""Draw all figures (light and dark variants) from results/results.pkl."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ROOT = Path(__file__).parent
FIG = ROOT / "results" / "fig"
FIG.mkdir(parents=True, exist_ok=True)
R = pickle.load(open(ROOT / "results" / "results.pkl", "rb"))

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"],
                  ramp=["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 s=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"],
                 ramp=["#1a1a19", "#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]),
}
T = None


def theme(name):
    global T
    T = THEMES[name]
    plt.rcParams.update({
        "figure.facecolor": T["bg"], "axes.facecolor": T["bg"], "savefig.facecolor": T["bg"],
        "text.color": T["ink"], "axes.labelcolor": T["ink2"], "axes.edgecolor": T["axis"],
        "xtick.color": T["muted"], "ytick.color": T["muted"], "xtick.labelcolor": T["ink2"],
        "ytick.labelcolor": T["ink2"], "grid.color": T["grid"], "grid.linewidth": 0.6,
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 5.5,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "axes.axisbelow": True,
    })


def cmap():
    return LinearSegmentedColormap.from_list("seq", T["ramp"])


def save(fig, name, th):
    fig.savefig(FIG / f"{name}_{th}.png", dpi=170, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def f1(th):
    d = R["fig2"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True, constrained_layout=True)
    for ax, kD in zip(axs, (1.0, 10.0)):
        m = ax.pcolormesh(d["lams"], d["lengths"], d[kD], cmap=cmap(), vmin=0, vmax=1, shading="gouraud")
        cs = ax.contour(d["lams"], d["lengths"], d[kD], levels=[0.022, 0.1, 0.3, 0.6], colors=[T["ink"]], linewidths=0.8)
        ax.clabel(cs, fmt="%g", fontsize=8)
        ax.axvline(0.1, color=T["s"][1], lw=1.2, ls=(0, (4, 3)))
        ax.set_xscale("log"); ax.grid(False)
        ax.set_title(f"reactieparameter $k_D$ = {kD:g}")
        ax.set_xlabel("initiatiesnelheid λ (1/s)")
    axs[0].set_ylabel("eiwitlengte (AA)")
    axs[0].text(0.115, 22, "gemeten λ ≈ 0,1/s", color=T["s"][1], fontsize=8.5)
    cb = fig.colorbar(m, ax=axs, pad=0.02, aspect=22)
    cb.set_label("co-translationele fractie α (vgl. 3)"); cb.outline.set_visible(False)
    save(fig, "f1_alpha_maps", th)


def f2(th):
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True, constrained_layout=True)
    for ax, lam in zip(axs, (0.1, 1.0)):
        s = R["kymo"][lam]
        t = np.arange(len(s)) * 0.25
        tt = np.repeat(t[:, None], 64, 1)
        pos, st = s[:, :, 0], s[:, :, 1]
        for val, col, lab in ((0, T["s"][0], "vrije keten"), (1, T["s"][1], "co-translationeel gedimeriseerd")):
            k = st == val
            ax.scatter(tt[k], pos[k], s=2.2, color=col, linewidths=0, label=lab)
        ax.set_title(f"λ = {lam:g}/s"); ax.set_xlabel("tijd (s)"); ax.set_xlim(0, 120); ax.set_ylim(0, 300)
    axs[0].set_ylabel("ribosoompositie (codon)")
    axs[0].legend(loc="upper left", markerscale=4, handletextpad=0.2, bbox_to_anchor=(0, 1.0))
    save(fig, "f2_kymograph", th)


def f3(th):
    sw = R["sweep"]; l = sw["lams"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True, constrained_layout=True)
    labs = {"A": "simulatie A: puntribosomen, alleen buren", "B": "B: + voetafdruk 10 codons, alle paren",
            "C": "C: + exittunnel 30 AA", "D": "D: + codonspecifieke snelheden"}
    for ax, kD in zip(axs, (1.0, 10.0)):
        ax.plot(l, sw[("mm", kD)], color=T["ink"], lw=2, label="paper vgl. 3 (per paar)")
        ax.plot(l, sw[("hazard", kD)], color=T["muted"], lw=2, ls=(0, (4, 2)), label="paar, hazard-vorm 1−e$^{-k_D⟨Ψ²⟩}$")
        for i, key in enumerate("ABCD"):
            a = np.array([r["alpha"] for r in sw[(key, kD)]])
            ax.plot(l, a, color=T["s"][i], marker="osD^"[i], lw=1.6, mec=T["bg"], mew=0.8, label=labs[key])
        ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_title(f"$k_D$ = {kD:g}, eiwit van 300 AA")
        ax.set_xlabel("initiatiesnelheid λ (1/s)")
    axs[0].set_ylabel("fractie ketens in co-translationeel dimeer")
    axs[0].legend(loc="upper left", fontsize=8)
    save(fig, "f3_sweep", th)


def f4(th):
    sw = R["sweep"]; i = int(np.argmin(abs(sw["lams"] - 0.1)))
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)
    for key, col in (("B", T["s"][1]), ("C", T["s"][2])):
        r = sw[(key, 1.0)][i]
        for ax, h in zip(axs, (r["hist_len"], r["hist_sep"])):
            hb = h[:300].reshape(30, 10).sum(1); hb = hb / hb.sum()
            ax.step(np.arange(30) * 10 + 5, hb, where="mid", color=col,
                    label={"B": "B: zonder tunnel", "C": "C: met exittunnel (30 AA)"}[key])
    axs[0].set_title("Lengte van de kortste keten bij dimerisatie"); axs[0].set_xlabel("blootgestelde ketenlengte (AA)")
    axs[1].set_title("Afstand tussen de twee ribosomen"); axs[1].set_xlabel("afstand (codons)")
    axs[0].set_ylabel("aandeel van dimerisaties"); axs[0].legend()
    save(fig, "f4_where", th)


def f5(th):
    p = R["polymer"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    ax = axs[0]
    ax.plot(p["ns"], p["gauss"] * 1e6, color=T["ink"], label="Gaussische formule")
    ax.plot(p["ns"], p["free"] * 1e6, "o", color=T["s"][0], mec=T["bg"], label="MC, zonder uitgesloten volume")
    ax.plot(p["ns"], p["parallel"] * 1e6, "s", color=T["s"][1], mec=T["bg"], label="MC, tunnels parallel")
    a = np.where(p["anti"] > 0, p["anti"], np.nan)
    ax.plot(p["ns"], a * 1e6, "D", color=T["s"][2], mec=T["bg"], label="MC, tunnels tegenovergesteld")
    ax.axhline(1.66e-3, color=T["muted"], lw=1, ls=(0, (4, 3)))
    ax.text(150, 2.2e-3, "1 molecuul per E. coli-cel (1,7 nM)", fontsize=8, color=T["ink2"])
    ax.set_yscale("log"); ax.set_ylim(1e-6, 30)
    ax.set_xlabel("blootgestelde ketenlengte (AA), beide ketens"); ax.set_ylabel("effectieve concentratie (µM)")
    ax.set_title("Lokale concentratie N-terminus, ribosomen 25 nm uiteen")
    ax.legend(fontsize=8, loc="upper left")
    ax = axs[1]
    for i, lam in enumerate(p["lams"]):
        ax.plot(p["kons"], p["alpha"][i], color=T["s"][i], marker="o", mec=T["bg"], label=f"λ = {lam:g}/s")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("associatieconstante $k_{on}$ (1/M/s)"); ax.set_ylabel("fractie ketens in co-translationeel dimeer")
    ax.set_title("Polymeermodel, eiwit van 300 AA"); ax.legend()
    save(fig, "f5_polymer", th)


def f6(th):
    r = R["rep"]
    fig = plt.figure(figsize=(9.2, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0:2, 0])
    k = r["alphas"] <= 0.06
    ax.plot(r["alphas"][k], r["eig"].real[k] * 1e4, color=T["s"][0], label="reëel deel")
    ax.plot(r["alphas"][k], abs(r["eig"].imag[k]) * 1e4, color=T["s"][1], ls=(0, (4, 2)), label="imaginair deel")
    ax.axhline(0, color=T["axis"], lw=1)
    ax.axvline(r["hopf"], color=T["muted"], lw=1, ls=(0, (2, 2)))
    ax.annotate(f"Hopf-punt\nα = {r['hopf']:.4f}".replace(".", ","), (r["hopf"], 0), (r["hopf"] + 0.006, 1.2),
                fontsize=9, color=T["ink"], arrowprops=dict(arrowstyle="-", color=T["muted"]))
    ax.set_xlabel("co-translationele fractie α"); ax.set_ylabel("leidende eigenwaarde (10⁻⁴ /s)")
    ax.set_title("Stabiliteit van het evenwicht"); ax.legend(loc="center right")
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(r["alphas"], r["hill"], color=T["s"][2])
    ax.set_xscale("symlog", linthresh=0.01); ax.set_ylim(1, 2.05)
    ax.axvline(r["hopf"], color=T["muted"], lw=1, ls=(0, (2, 2)))
    ax.set_xlabel("α"); ax.set_ylabel("effectieve Hill-exponent"); ax.set_title("Niet-lineariteit dimeer ↔ promotoractiviteit")
    for i, a in enumerate((0.0, 0.02, 0.1)):
        ax = fig.add_subplot(gs[i, 1])
        t, y = r["tc"][a]
        for j, nm in enumerate(("promotor 1", "promotor 2", "promotor 3")):
            ax.plot(t / 3600, y[j], color=T["s"][j], lw=1.4, label=nm)
        ax.set_xlim(0, 50); ax.set_ylim(0, 0.62); ax.set_title(f"α = {a:g}".replace(".", ","))
        ax.set_ylabel("promotoractiviteit")
        if i == 0: ax.legend(ncol=3, loc="upper right", fontsize=8)
        if i == 2: ax.set_xlabel("tijd (uur)")
    save(fig, "f6_repressilator", th)


def f7(th):
    d, r = R["design"], R["rep"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True, width_ratios=[1.25, 1])
    ax = axs[0]
    m = ax.pcolormesh(d["lams"], d["lengths"], d["alpha"], cmap=cmap(), vmin=0, vmax=1, shading="gouraud")
    cs = ax.contour(d["lams"], d["lengths"], d["alpha"], levels=[r["hopf"]], colors=[T["s"][1]], linewidths=2)
    ax.clabel(cs, fmt="α = 0,022", fontsize=8.5)
    ax.axvline(0.1, color=T["ink"], lw=1, ls=(0, (4, 3)))
    ax.text(0.115, 385, "gemeten λ", fontsize=8.5, color=T["ink"])
    ax.set_xscale("log"); ax.grid(False)
    ax.set_xlabel("initiatiesnelheid λ (1/s)  ← zwakkere RBS"); ax.set_ylabel("eiwitlengte (AA)")
    ax.set_title("Ontwerpkaart: gesimuleerde α (variant C, $k_D$ = 1)")
    cb = fig.colorbar(m, ax=ax, pad=0.02); cb.outline.set_visible(False); cb.set_label("α")
    ax = axs[1]
    ax.plot(r["kdim"], r["hopf_kdim"], color=T["s"][0], marker="o", ms=4, mec=T["bg"])
    ax.plot([1e-8], [r["hopf"]], "o", ms=9, mfc="none", mec=T["s"][1], mew=2)
    ax.annotate("waarde in de paper", (1e-8, r["hopf"]), (1.5e-10, 0.06), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=T["muted"]))
    ax.set_xscale("log"); ax.set_ylim(0, 0.085)
    ax.set_xlabel("cytosolische dimerisatiesnelheid $k_{dim}$"); ax.set_ylabel("kritische α (Hopf-punt)")
    ax.set_title("Oscillaties alleen onder de curve")
    save(fig, "f7_design", th)


def f8(th):
    tog = R["toggle"]
    fig, axs = plt.subplots(3, 3, figsize=(9.2, 5.6), sharex=True, constrained_layout=True)
    for i, a in enumerate((0.02, 0.1, 0.5)):
        for j, d in enumerate((2.0, 3.0, 7.0)):
            ax = axs[i, j]; r = tog[(True, d, a)]
            ax.axvspan(470, 530, color=T["s"][0], alpha=0.25, lw=0); ax.axvspan(1470, 1530, color=T["s"][1], alpha=0.25, lw=0)
            ax.plot(r["t"], r["G1"], color=T["s"][0], lw=1.6, label="repressor 1")
            ax.plot(r["t"], r["G2"], color=T["s"][1], lw=1.6, label="repressor 2")
            ax.set_title(f"α = {a:g}, afstand 10$^{{{int(d)}}}$ bp   " + ("bistabiel" if r["bistable"] else "geen schakelaar"),
                         fontsize=8.5, color=T["ink"] if r["bistable"] else T["ink2"])
            ax.set_xlim(0, 4000)
            if j == 0: ax.set_ylabel("dimeer (a.u.)")
            if i == 2: ax.set_xlabel("tijd (min)")
    axs[0, 0].legend(fontsize=8, loc="upper right")
    save(fig, "f8_toggle", th)

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True, constrained_layout=True)
    bm = ListedColormap([T["ramp"][1] if th == "light" else T["ramp"][1], T["ramp"][3]])
    for ax, lit, ttl in zip(axs, (True, False), ("Vergelijkingen zoals in ToggleSwitch.nb", "Index-consistente lezing")):
        s = R[("toggle_scan", lit)]
        z = (s["ratio"] > 2).astype(float)
        x = np.arange(len(s["alphas"]) + 1); y = np.arange(len(s["ds"]) + 1)
        ax.pcolormesh(x, y, z, cmap=bm, vmin=0, vmax=1, edgecolors=T["bg"], linewidth=1.5)
        ax.set_xticks(x[:-1] + 0.5, [f"{a:g}" for a in s["alphas"]], rotation=60, fontsize=8)
        ax.set_yticks(y[:-1][::2] + 0.5, [f"{d:g}" for d in s["ds"][::2]])
        ax.grid(False); ax.set_title(ttl); ax.set_xlabel("α")
    axs[0].set_ylabel("log₁₀ afstand tussen genen (bp)")
    from matplotlib.patches import Patch
    axs[1].legend(handles=[Patch(color=T["ramp"][3], label="bistabiel"), Patch(color=T["ramp"][1], label="geen schakelaar")],
                  loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "f9_toggle_scan", th)


if __name__ == "__main__":
    for th in ("light", "dark"):
        theme(th)
        for f in (f1, f2, f3, f4, f5, f6, f7, f8):
            f(th)
    print("figures written to", FIG)
