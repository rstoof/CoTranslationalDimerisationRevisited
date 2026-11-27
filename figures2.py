"""Figures for the experiments that test the theory against data (light and dark variants)."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

import figures as F
from ctd import disp

ROOT = Path(__file__).parent
R = pickle.load(open(ROOT / "results" / "tests.pkl", "rb"))
G = pickle.load(open(ROOT / "results" / "geometry.pkl", "rb"))
K = pickle.load(open(ROOT / "results" / "kernel.pkl", "rb"))
ONE_PER_CELL = 1.66e-9  # mol/l: one molecule in an E. coli cell of 1 fl


def T():
    return F.T


def feature_bar(ax, feats, y=1.04, h=0.045):
    """Domain map above the axes (data x, axes-fraction y)."""
    tr = ax.get_xaxis_transform()
    for i, (name, a, b) in enumerate(feats):
        ax.add_patch(Rectangle((a, y), b - a, h, transform=tr, clip_on=False, fc=T()["grid"], ec=T()["axis"], lw=0.8))
        ax.text((a + b) / 2, y + h / 2, name, transform=tr, ha="center", va="center", fontsize=7.5, color=T()["ink2"])


def g1(th):
    E = R["ecoli"]; C = disp.CONSTRUCTS
    fig, axs = plt.subplots(2, 2, figsize=(9.4, 6.6), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.14)
    ax = axs[0, 0]
    for rep, ls in (("1", "-"), ("2", (0, (4, 2)))):
        r = E[("1B", rep)]; ax.plot(r["x"], r["E"], color=T()["s"][0], ls=ls, lw=1.8, label=f"coiled coil, replicaat {rep}")
        r = E[("1Bmut", rep)]; ax.plot(r["x"], r["E"], color=T()["s"][1], ls=ls, lw=1.8, label=f"mutant zonder coiled coil, replicaat {rep}")
    ax.set_yscale("log"); ax.set_ylim(0.4, 60); ax.set_xlim(0, 395)
    ax.legend(fontsize=7.5, loc="upper left"); feature_bar(ax, C["1B"]["features"])
    ax.set_title("Lamine coil 1B gefuseerd aan mCherry", y=1.11); ax.set_ylabel("disoom / monosoom (× t.o.v. begin)")
    ax = axs[0, 1]
    for rep, ls in (("1", "-"), ("2", (0, (4, 2)))):
        r = E[("LMNC", rep)]; ax.plot(r["x"], r["E"], color=T()["s"][0], ls=ls, lw=1.8, label=f"replicaat {rep}")
    ax.set_yscale("log"); ax.set_ylim(0.4, 60); ax.set_xlim(0, 587)
    ax.legend(fontsize=7.5, loc="lower right"); feature_bar(ax, C["LMNC"]["features"])
    ax.set_title("Lamine C (residu 31–572)", y=1.11)
    ax = axs[1, 0]
    r = E[("DCTN1", "1")]
    x20 = r["x"][:len(r["x"]) // 2 * 2].reshape(-1, 2).mean(1); e20 = r["E"][:len(r["E"]) // 2 * 2].reshape(-1, 2).mean(1)
    ax.plot(x20, e20, color=T()["s"][0], lw=1.8)
    ax.set_yscale("log"); ax.set_ylim(0.4, 60); ax.set_xlim(0, 1325)
    feature_bar(ax, C["DCTN1"]["features"]); ax.set_title("Dynactine p150 (DCTN1)", y=1.11)
    ax.set_xlabel("codon"); ax.set_ylabel("disoom / monosoom (× t.o.v. begin)")
    ax = axs[1, 1]
    for rep, mk, col in (("1", "o", T()["s"][0]), ("2", "s", T()["s"][2])):
        r = E[("1B", rep)]
        ax.plot(r["x"], r["f"], color=col, lw=1.8, label=f"replicaat {rep}: plateau {r['f_plateau']:.2f}".replace(".", ","))
        ax.plot(r["x"][19:], r["f_dep"][19:], mk, color=col, ms=3.5, mec=T()["bg"], mew=0.5)
    ax.axhline(0, color=T()["axis"], lw=1); ax.set_ylim(-0.08, 1); ax.set_xlim(0, 395)
    ax.legend(fontsize=8, loc="upper left"); feature_bar(ax, C["1B"]["features"])
    ax.set_title("Fractie ribosomen in een keten-gekoppeld paar", y=1.11); ax.set_xlabel("codon"); ax.set_ylabel("fractie f")
    F.save(fig, "g1_ecoli", th)


def g2(th):
    f = R["fit"]; x = f["x"]
    E = R["ecoli"]
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.0), constrained_layout=True, width_ratios=[2.3, 1])
    ax = axs[0]
    lo = np.minimum(E[("1B", "1")]["f"][:39], E[("1B", "2")]["f"][:39]); hi = np.maximum(E[("1B", "1")]["f"][:39], E[("1B", "2")]["f"][:39])
    ax.vlines(x, lo, hi, color=T()["muted"], lw=1.2)
    ax.plot(x, f["target"], "o", color=T()["ink"], ms=4, label="gemeten (gemiddelde van 2 replicaten)", zorder=5)
    names = {"paper": "bollenmodel uit de paper", "gated_paper": "bollenmodel + interface-poort",
             "constant": "poort + constante paarsnelheid", "structure": "poort + structuurgebaseerde kern"}
    for i, k in enumerate(("paper", "gated_paper", "constant", "structure")):
        ax.plot(x, f[k]["f"], color=T()["s"][i], lw=2, ls=(0, (4, 2)) if k == "paper" else "-", label=names[k])
    ax.axvspan(340, 395, color=T()["grid"], alpha=0.6, lw=0)
    ax.text(367, 0.04, "niet\ngefit", ha="center", fontsize=7.5, color=T()["ink2"])
    ax.set_xlim(0, 395); ax.set_ylim(-0.03, 1.0); ax.set_xlabel("codon"); ax.set_ylabel("fractie ribosomen in gekoppeld paar")
    ax.legend(fontsize=8, loc="upper left"); feature_bar(ax, disp.CONSTRUCTS["1B"]["features"])
    ax.set_title("Coil 1B–mCherry in E. coli: meting en vier modellen", y=1.11)
    ax = axs[1]
    ks = ("paper", "gated_paper", "constant", "structure")
    v = [f[k]["rmse"] for k in ks]
    ax.barh(range(4), v, color=[T()["s"][i] for i in range(4)], height=0.55)
    for i, val in enumerate(v):
        ax.text(val + 0.004, i, f"{val:.3f}".replace(".", ","), va="center", fontsize=8.5, color=T()["ink"])
    ax.set_yticks(range(4), ["bollen", "bollen\n+ poort", "poort\n+ constant", "poort\n+ structuur"], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, max(v) * 1.25); ax.grid(axis="y", visible=False)
    ax.set_xlabel("afwijking (RMSE in f)"); ax.set_title("Hoe goed past het?")
    F.save(fig, "g2_fit", th)


_DS = None


def g3(th):
    global _DS
    from ctd import pairgeom as pg
    if _DS is None:
        _DS = pg.Disome()
    d = _DS
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 4.1), constrained_layout=True, width_ratios=[1.15, 1, 1])
    # A: projection of the pair onto the plane of the two exits and the mean tunnel direction
    ax = axs[0]
    e1 = (d.exit2 - d.exit1); dist = np.linalg.norm(e1); e1 /= dist
    um = d.u1 + d.u2; e2 = um - (um @ e1) * e1; e2 /= np.linalg.norm(e2)
    o = (d.exit1 + d.exit2) / 2
    pr = lambda p: np.stack([(p - o) @ e1, (p - o) @ e2], -1) / 10.0       # nm
    back = np.isin(d.at["atom"], ["P", "CA"])
    for k, col in ((1, T()["s"][0]), (2, T()["s"][1])):
        p = pr(d.xyz[(d.rib == k) & back][::4]); ax.scatter(p[:, 0], p[:, 1], s=1.2, color=col, alpha=0.35, linewidths=0)
    m = pr(d.mrna); ax.plot(m[:, 0], m[:, 1], color=T()["ink"], lw=1.6)
    for ex, u, lab, col in ((d.exit1, d.u1, "uitgang 1", T()["s"][0]), (d.exit2, d.u2, "uitgang 2", T()["s"][1])):
        p = pr(ex); q = pr(ex + 70 * u)
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=T()["ink"], lw=1.6))
        ax.plot(*p, "o", color=T()["bg"], mec=T()["ink"], mew=1.5, ms=7, zorder=5)
    p1, p2 = pr(d.exit1), pr(d.exit2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=T()["ink"], lw=1, ls=(0, (3, 3)))
    ax.text(0, p1[1] + 1.6, f"{dist / 10:.0f} nm".replace(".", ","), ha="center", fontsize=9, color=T()["ink"],
            bbox=dict(fc=T()["bg"], ec="none", pad=1.5))
    ax.text(0, -14.5, "mRNA", ha="center", fontsize=8, color=T()["ink"])
    ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("nm"); ax.set_ylabel("nm")
    ax.set_xlim(-28, 28); ax.set_ylim(-34, 10)
    ax.set_title("Twee botsende E. coli-ribosomen")
    ax.legend(handles=[Patch(color=T()["s"][0], label="voorste ribosoom"), Patch(color=T()["s"][1], label="achterste ribosoom")],
              fontsize=7.5, loc="lower center", ncol=2, handlelength=1)
    # B: effective concentration versus exposed chain length
    ax = axs[1]
    n = K["n_grid"]
    ok = G["direct"] > 0
    ax.plot(G["n_trail"][ok], G["direct"][ok] * 1e6, "o-", color=T()["ink"], lw=1.8, ms=4, label="botsend paar (structuur)")
    for i, e in enumerate((5, 20, 80, 320)):
        ax.plot(n, np.diag(K["ensemble"][int(e)]["table"]) * 1e6, color=T()["s"][i], lw=1.8, label=f"+{e} codons, vrije oriëntatie")
    ax.axhline(ONE_PER_CELL * 1e6, color=T()["muted"], lw=1, ls=(0, (4, 3)))
    ax.text(545, ONE_PER_CELL * 1e6 * 1.5, "1 molecuul per cel", ha="right", fontsize=7.5, color=T()["ink2"])
    ax.axvspan(105, 120, color=T()["grid"], alpha=0.8, lw=0)
    ax.text(124, 14, "gemeten start", ha="left", va="center", fontsize=7.5, color=T()["ink2"])
    ax.set_yscale("log"); ax.set_ylim(1e-5, 30); ax.set_xlim(0, 550)
    ax.set_xlabel("blootgestelde residuen per keten"); ax.set_ylabel("effectieve concentratie (µM)")
    ax.set_title("Lokale concentratie van de partner"); ax.legend(fontsize=7.2, loc="lower right")
    # C: versus spacing
    ax = axs[2]
    ex = np.concatenate([[0], K["extras"]]) + G["s_min"]
    for i, nn in enumerate((60, 100, 200)):
        j = list(n).index(nn)
        ax.plot(ex, K["table"][j, j, :] * 1e6, "o-", color=T()["s"][i], lw=1.8, ms=4, mec=T()["bg"], label=f"{nn} residuen")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-5, 30)
    ax.set_xlabel("afstand tussen de ribosomen (codons)"); ax.set_title("Afhankelijkheid van de afstand")
    ax.legend(fontsize=8, loc="lower right", title="blootgesteld", title_fontsize=8)
    F.save(fig, "g3_geometry", th)


def g4(th):
    H = R["human"]
    fig, axs = plt.subplots(2, 2, figsize=(9.4, 6.4), constrained_layout=True)
    ax = axs[0, 0]
    bins = np.logspace(np.log10(40), np.log10(2600), 36)
    ax.hist(H["onset"], bins=bins, color=T()["s"][0], ec=T()["bg"], lw=0.6)
    lo = 30 + G["free_path"] / 2 / 3.6
    ax.axvline(lo, color=T()["s"][1], lw=1.4, ls=(0, (4, 2)))
    ax.text(lo * 0.96, ax.get_ylim()[1] * 0.97, "minimum voor\nbotsende buren", fontsize=7.5, color=T()["ink"], va="top", ha="right")
    ax.set_xscale("log"); ax.set_xlabel("start van de koppeling (codon)"); ax.set_ylabel("aantal eiwitten")
    ax.set_title(f"Start bij {H['n_high']} humane eiwitten (mediaan codon {np.median(H['onset']):.0f})")
    ax = axs[0, 1]
    m = H["onset_meta"]
    ax.plot(m["x"], m["enrich"], color=T()["s"][0], lw=2)
    ax.axvline(0, color=T()["muted"], lw=1, ls=(0, (2, 2))); ax.axhline(1, color=T()["axis"], lw=1)
    ax.set_xlabel("codons vanaf de start"); ax.set_ylabel("disoom / monosoom (×)"); ax.set_title("Na de start loopt de koppeling in ~50 codons op")
    ax = axs[1, 0]
    s_ = H["stop_meta"]
    ax.fill_between(s_["high"]["x"], s_["high"]["ci"][0], s_["high"]["ci"][1], color=T()["s"][0], alpha=0.22, lw=0)
    ax.plot(s_["high"]["x"], s_["high"]["enrich"], color=T()["s"][0], lw=2, label=f"co-co-kandidaten (n = {s_['high']['n']})")
    ax.plot(s_["control"]["x"], s_["control"]["enrich"], color=T()["s"][1], lw=2, label=f"controle-eiwitten (n = {s_['control']['n']})")
    ax.axhline(1, color=T()["axis"], lw=1); ax.set_ylim(0.5, 1.2)
    ax.set_xlabel("codons vóór het stopcodon"); ax.set_ylabel("disoom / monosoom (t.o.v. −450…−300)")
    ax.set_title("Richting het einde raakt een deel zijn partner kwijt"); ax.legend(fontsize=8, loc="lower left")
    ax = axs[1, 1]
    e = H["eff"]
    ax.scatter(e["remaining"], e["eff"], s=6, color=T()["s"][0], alpha=0.35, linewidths=0)
    q = np.quantile(e["remaining"], np.linspace(0, 1, 7))
    xm = [np.median(e["remaining"][(e["remaining"] >= a) & (e["remaining"] <= b)]) for a, b in zip(q[:-1], q[1:])]
    ym = [np.median(e["eff"][(e["remaining"] >= a) & (e["remaining"] <= b)]) for a, b in zip(q[:-1], q[1:])]
    ax.plot(xm, ym, "o-", color=T()["ink"], lw=1.8, ms=5, label="mediaan per zesde")
    rho = e["rho_remaining"]
    ax.set_xscale("log"); ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("codons na de start (L − start)"); ax.set_ylabel("efficiëntie (monosoom-afname)")
    ax.set_title(f"Lengte na de start doet weinig (Spearman ρ = {rho[0]:.2f})".replace(".", ","))
    ax.legend(fontsize=8, loc="upper left")
    F.save(fig, "g4_human", th)


def g5(th):
    L = R["lambda"]
    fig, ax = plt.subplots(figsize=(6.6, 3.9), constrained_layout=True)
    ax.plot(L["lams"], L["paper"], color=T()["ink"], lw=2, ls=(0, (4, 2)), label="paper, vgl. 3 ($k_D$ = 1), per paar")
    labs = {"no gate, kon 1e6": "structuur, geen poort, $k_{on}$ = 10⁶", "gate 100, kon 1e6": "structuur, interface na 100 AA, $k_{on}$ = 10⁶",
            "gate 100, kon 1e5": "structuur, interface na 100 AA, $k_{on}$ = 10⁵"}
    for i, k in enumerate(("no gate, kon 1e6", "gate 100, kon 1e6", "gate 100, kon 1e5")):
        ax.plot(L["lams"], L[k]["alpha"], "o-", color=T()["s"][i], lw=1.8, ms=4, mec=T()["bg"], label=labs[k])
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("initiatiesnelheid λ (1/s), eiwit van 400 AA, 15 codons/s"); ax.set_ylabel("fractie ketens in co-translationeel dimeer")
    ax.legend(fontsize=8, loc="upper left")
    F.save(fig, "g5_lambda", th)


def g6(th):
    Lc = R["locus"]; k = Lc["ktx"]
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.9), constrained_layout=True)
    lab = "N-terminal interface"
    seen = lambda r, key: np.array([q[key] if q[key] * q["n"] >= 6 else np.nan for q in r])   # at least 3 pairs observed
    ax = axs[0]
    r = Lc[(lab, "territory 0.2 um")]
    for i, (key, nm) in enumerate((("cis", "zelfde mRNA"), ("sibling", "naburig transcript aan het DNA"), ("territory", "ander transcript bij het gen"))):
        ax.plot(k, seen(r, key), "o-", color=T()["s"][i], lw=1.8, ms=4, mec=T()["bg"], label=nm)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-4, 1.5)
    ax.set_xlabel("transcriptie-initiatie (1/s)"); ax.set_ylabel("fractie van alle ketens")
    ax.set_title("Wie is de partner?"); ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.62))
    ax = axs[1]
    for i, (terr, nm) in enumerate((("DNA-tethered only", "alleen aan het DNA"), ("territory 0.2 um", "mRNA blijft binnen 0,2 µm"), ("whole cell", "mRNA verspreid over de cel"))):
        r = Lc[(lab, terr)]
        y = np.array([q["sibling"] + q["territory"] if (q["sibling"] + q["territory"]) * q["n"] >= 6 else np.nan for q in r])
        ax.plot(k, y, "o-", color=T()["s"][i + 3], lw=1.8, ms=4, mec=T()["bg"], label=nm)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-4, 1.5)
    ax.set_xlabel("transcriptie-initiatie (1/s)"); ax.set_title("Paren tussen transcripten"); ax.legend(fontsize=7.5, loc="upper left")
    ax = axs[2]
    r = Lc[(lab, "territory 0.2 um")]
    tx = np.array([q["tx_rate"] for q in r]); prot = np.array([q["protein_rate"] for q in r])
    cis = np.array([q["cis"] for q in r]) * prot / 2
    trans = seen(r, "sibling") * prot / 2 + np.array([q["territory"] for q in r]) * prot / 2
    ok = np.isfinite(trans)
    sl = lambda y, m: np.polyfit(np.log(tx[m]), np.log(y[m]), 1)[0]
    ax.plot(tx, cis, "o-", color=T()["s"][0], lw=1.8, ms=4, mec=T()["bg"], label=f"binnen één mRNA (helling {sl(cis, tx > 0):.2f})".replace(".", ","))
    ax.plot(tx[ok], trans[ok], "o-", color=T()["s"][1], lw=1.8, ms=4, mec=T()["bg"], label=f"tussen transcripten (helling {sl(trans, ok):.2f})".replace(".", ","))
    b = Lc["bursty"]
    btx = np.array([q["tx_rate"] for q in b["res"]]); bpr = np.array([q["protein_rate"] for q in b["res"]])
    btr = np.array([q["sibling"] + q["territory"] for q in b["res"]]) * bpr / 2
    bs = np.polyfit(np.log(btx), np.log(btr), 1)[0]
    ax.plot(btx, btr, "s--", color=T()["s"][2], lw=1.5, ms=5, mec=T()["bg"], label=f"idem, promotor schakelt aan/uit (helling {bs:.2f})".replace(".", ","))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(top=60)
    ax.set_xlabel("transcripten per seconde"); ax.set_ylabel("co-translationele dimeren per seconde")
    ax.set_title("Lineair of kwadratisch?"); ax.legend(fontsize=7.2, loc="upper left")
    F.save(fig, "g6_locus", th)


def g7(th):
    P = R["repressors"]
    H = R["human"]
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True, width_ratios=[1.25, 1])
    ax = axs[0]
    names = ["TetR(B)", "LacI", "lambda cI"]
    for i, nm in enumerate(names):
        p = P[nm]; L = p["L"]
        ax.add_patch(Rectangle((0, i - 0.28), 1, 0.56, fc=T()["bg"], ec=T()["axis"], lw=1))
        ax.add_patch(Rectangle((1 - 30 / L, i - 0.28), 30 / L, 0.56, fc=T()["grid"], ec="none"))
        ax.vlines(p["residues"] / L, i - 0.26, i + 0.26, color=T()["s"][0], lw=1.1)
        ax.plot([p["q50"] / L], [i - 0.4], "v", color=T()["ink"], ms=6)
    fr = (H["onset"] - 30) / H["length"]
    q = np.percentile(fr, [10, 25, 50, 75, 90])
    i = 3
    ax.add_patch(Rectangle((q[1], i - 0.2), q[3] - q[1], 0.4, fc=T()["s"][1], ec="none", alpha=0.75))
    ax.plot([q[0], q[4]], [i, i], color=T()["s"][1], lw=1.5); ax.plot([q[2], q[2]], [i - 0.2, i + 0.2], color=T()["ink"], lw=1.8)
    ax.set_yticks(range(4), ["TetR (207 AA)", "LacI (360 AA)", "λ cI (237 AA)", "co-co-assemblers\n(829, mens)"], fontsize=8.5)
    ax.set_ylim(3.6, -0.75); ax.set_xlim(0, 1); ax.grid(False)
    ax.set_xlabel("positie in het eiwit (fractie van de lengte)")
    ax.set_title("Waar ligt de dimeer-interface?")
    ax.text(0.99, -0.55, "grijs: zit bij loslaten\nnog in de tunnel", ha="right", va="center", fontsize=7.5, color=T()["ink2"])
    ax = axs[1]
    win = 300 - 30 - P["scan_n_on"]
    sty = {(0.1, 1e6): (T()["s"][0], "-"), (1.0, 1e6): (T()["s"][1], "-"), (0.1, 1e7): (T()["s"][0], (0, (4, 2))), (1.0, 1e7): (T()["s"][1], (0, (4, 2)))}
    for kon in (1e6, 1e7):
        for j, lam in enumerate(P["scan_lams"]):
            col, ls = sty[(lam, kon)]
            ax.plot(win, np.maximum(P[("scan", kon)][j], 1e-4), color=col, ls=ls, lw=1.8, marker="o", ms=3.5, mec=T()["bg"],
                    label=f"λ = {lam:g}/s, $k_{{on}}$ = 10$^{int(np.log10(kon))}$".replace(".", ","))
    ax.axhline(0.022, color=T()["ink"], lw=1, ls=(0, (2, 2)))
    ax.text(268, 0.026, "Hopf-punt repressilator (0,022)", fontsize=7.5, color=T()["ink"], ha="right")
    for nm, short, dy in (("TetR(B)", "TetR", 0), ("LacI", "LacI", 0), ("lambda cI", "cI", 0)):
        w = P[nm]["window50"]
        ax.axvline(w, color=T()["muted"], lw=1, ls=(0, (1, 2)))
        ax.text(w + 3, 1.6e-4, f"{short}\n({w})", fontsize=7.5, color=T()["ink"], va="bottom")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5); ax.set_xlim(0, 272)
    ax.set_xlabel("codons tussen 'interface buiten de tunnel' en stop"); ax.set_ylabel("co-translationele fractie α")
    ax.set_title("Eiwit van 300 AA"); ax.legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.03))
    F.save(fig, "g7_repressors", th)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["g1", "g2", "g3", "g4", "g5", "g6", "g7"]
    for th in ("light", "dark"):
        F.theme(th)
        for name in which:
            globals()[name](th)
    print("figures written")
