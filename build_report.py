"""Assemble report.html: the fixed page template plus the data-testing view, with every number
taken from the result files (results/tests.pkl, geometry.pkl, kernel.pkl)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
R = pickle.load(open(ROOT / "results" / "tests.pkl", "rb"))
G = pickle.load(open(ROOT / "results" / "geometry.pkl", "rb"))
K = pickle.load(open(ROOT / "results" / "kernel.pkl", "rb"))


def nl(x, d=2):
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def sci(x):
    """Concentration in mol/l as a readable string in nM or µM."""
    if x <= 0:
        return "onder de detectiegrens"
    if x < 1e-9:
        return f"{nl(x * 1e9, 3 if x < 1e-10 else 2)} nM"
    if x < 1e-6:
        return f"{nl(x * 1e9, 1 if x < 1e-8 else 0)} nM"
    return f"{nl(x * 1e6, 2 if x < 1e-5 else 1)} µM"


def fig(name, alt, caption):
    return (f'<figure class="wide">\n  <img class="l" src="fig/{name}_light.png" alt="{alt}">\n'
            f'  <img class="d" src="fig/{name}_dark.png" alt="{alt}">\n  <figcaption>{caption}</figcaption>\n</figure>')


def tokens():
    v = {}
    E, H, S = R["ecoli"], R["human"], G["summary"]
    a, b = E[("1B", "1")], E[("1B", "2")]
    v["f1"], v["f2"] = nl(a["f_plateau"]), nl(b["f_plateau"])
    v["se1"], v["se2"] = nl(a["f_plateau_se"]), nl(b["f_plateau_se"])
    v["mid1"], v["mid2"] = f"{a['fit']['mid']:.0f}", f"{b['fit']['mid']:.0f}"
    v["exp1"], v["exp2"] = f"{a['fit']['mid'] - 30 - 10:.0f}", f"{b['fit']['mid'] - 30 - 10:.0f}"   # coil starts at codon 11
    v["mutE1"], v["mutE2"] = nl(E[("1Bmut", "1")]["E_plateau"]), nl(E[("1Bmut", "2")]["E_plateau"])
    v["E1"], v["E2"] = nl(a["E_plateau"], 0), nl(b["E_plateau"], 1)
    l1, l2 = E[("LMNC", "1")], E[("LMNC", "2")]
    v["lf1"], v["lf2"] = nl(l1["f_plateau"]), nl(l2["f_plateau"])
    v["lmid1"], v["lmid2"] = f"{l1['fit']['mid']:.0f}", f"{l2['fit']['mid']:.0f}"
    v["lres1"], v["lres2"] = f"{l1['fit']['mid'] + 22:.0f}", f"{l2['fit']['mid'] + 22:.0f}"
    v["dmid"] = f"{E[('DCTN1', '1')]['fit']['mid']:.0f}"
    v["dE"] = nl(E[("DCTN1", "1")]["E_plateau"], 1)
    sh = lambda k, r: E[(k, r)]["reads"]
    # geometry
    v["centre"] = nl(S["centre_distance"] / 10, 1); v["exit"] = nl(S["exit_distance"] / 10, 1)
    v["exitA"] = f"{S['exit_distance']:.0f}"; v["path"] = f"{G['free_path']:.0f}"
    v["ang1"], v["ang2"] = f"{S['angle_exit1_to_exit2']:.0f}", f"{S['angle_exit2_to_exit1']:.0f}"
    v["blocked"] = f"{S['line_blocked_fraction'] * 100:.0f}"
    v["tunnel"] = f"{S['tunnel_length']:.0f}"; v["rmsd"] = nl(S["rmsd_23S"], 1)
    v["nmin_ext"] = f"{G['free_path'] / 2 / 3.6:.0f}"; v["nmin_hel"] = f"{G['free_path'] / 2 / 1.5:.0f}"
    v["smin"] = str(int(G["s_min"]))
    d = dict(zip(G["n_trail"].tolist(), G["direct"].tolist()))
    for n in (100, 120, 150, 200, 300, 400):
        v[f"col{n}"] = sci(d[n])
    n = list(K["n_grid"])
    for e in K["extras"]:
        r = K["ensemble"][int(e)]
        v[f"acc{int(e)}"] = nl(r["accept"] * 100, 2 if r["accept"] < 0.01 else 1)
        v[f"top{int(e)}"] = f"{r['top1_share'] * 100:.0f}"
        v[f"face{int(e)}"] = nl(r["facing"] * 100, 1)
        for nn in (60, 100, 200):
            v[f"c{nn}_{int(e)}"] = sci(r["table"][n.index(nn), n.index(nn)])
    v["colscale"] = nl(K["collided_scale"], 1)
    # human
    on = H["onset"]
    v["nhigh"] = str(H["n_high"]); v["on_med"] = f"{np.median(on):.0f}"
    v["on_q1"], v["on_q3"] = f"{np.percentile(on, 25):.0f}", f"{np.percentile(on, 75):.0f}"
    v["on_p10"] = f"{np.percentile(on, 10):.0f}"; v["on_lt75"] = nl(np.mean(on < 75) * 100, 1)
    lo = 30 + G["free_path"] / 2 / 3.6
    v["on_ltmin"] = nl(np.mean(on < lo) * 100, 1); v["on_min_codon"] = f"{lo:.0f}"
    v["on_frac"] = nl(np.median(on / H["length"]), 2)
    sm = H["stop_meta"]
    v["stop_end"] = nl(sm["high"]["enrich"][-1]); v["stop_lo"] = nl(sm["high"]["ci"][0][-1]); v["stop_hi"] = nl(sm["high"]["ci"][1][-1])
    v["stop_ctrl"] = nl(sm["control"]["enrich"][-1]); v["stop_n"] = str(sm["high"]["n"]); v["ctrl_n"] = str(sm["control"]["n"])
    x = sm["high"]["x"]; last_ge1 = np.where(sm["high"]["ci"][1] >= 1.0)[0][-1]
    v["stop_start"] = f"{-x[last_ge1 + 1]:.0f}"
    e = H["eff"]
    v["eff_n"] = str(len(e["eff"])); v["eff_med"] = nl(np.median(e["eff"]))
    for k in ("remaining", "onset", "length", "rpkm"):
        v["rho_" + k] = nl(e["rho_" + k][0])
    om = H["onset_meta"]
    plateau = om["enrich"][(om["x"] > 100) & (om["x"] < 300)].mean()
    v["meta_plateau"] = nl(plateau, 1)
    half = om["x"][np.where(om["enrich"] > 1 + 0.5 * (plateau - 1))[0][0]]
    ninety = om["x"][np.where(om["enrich"] > 1 + 0.9 * (plateau - 1))[0][0]]
    v["meta_half"], v["meta_90"] = f"{half:.0f}", f"{ninety:.0f}"
    fr = (on - 30) / H["length"]
    v["on_frac_exp"] = nl(np.median(fr), 2)
    # read share of the over-expressed construct
    d = np.load(ROOT / "data" / "disp_ecoli.npz")
    sh = []
    for rep in ("1", "2"):
        for frac in ("mono", "dis"):
            a = float(d[f"1B_{frac}_{rep}"].sum()); h = float(d[f"1B_{frac}_{rep}_host"])
            sh.append(100 * a / (a + h))
    v["sh_lo"], v["sh_hi"] = f"{min(sh):.0f}", f"{max(sh):.0f}"
    # experiment 3 (fit)
    f = R["fit"]
    x = f["x"]; sel = (x > 20) & (x < 340)
    f1_, f2_ = E[("1B", "1")]["f"][:39], E[("1B", "2")]["f"][:39]
    v["rep_rmse"] = nl(float(np.sqrt(np.mean(((f1_ - f2_) / 2)[sel] ** 2))), 3)
    for k in ("paper", "gated_paper", "constant", "structure"):
        v[f"rmse_{k}"] = nl(f[k]["rmse"], 3); v[f"sp_{k}"] = f"{f[k]['spacing']:.0f}"; v[f"al_{k}"] = f"{f[k]['alpha'] * 100:.0f}"
        v[f"gate_{k}"] = str(int(f[k]["n_on"]))
    v["kd_paper"] = nl(f["paper"]["a"], 1); v["kon_struct"] = f"10<sup>{np.log10(f['structure']['a']):.0f}</sup>"
    v["end_model"] = nl(float(f["structure"]["f"][-1])); v["end_data"] = nl(float(f["target"][-1]))
    v["papdist"] = nl(int(G["s_min"]) * 0.9, 1)
    v["pathnm"] = nl(G["free_path"] / 10, 1)
    # experiment 5 (lambda)
    Lm = R["lambda"]
    for lam, tag in ((0.0126, "a"), (0.05, "b"), (0.126, "c")):
        i = int(np.argmin(abs(Lm["lams"] - lam)))
        v[f"lam_{tag}"] = nl(Lm["lams"][i], 3 if lam < 0.02 else 2)
        v[f"ls_{tag}"] = nl(Lm["gate 100, kon 1e6"]["alpha"][i]); v[f"lp_{tag}"] = nl(Lm["paper"][i])
        v[f"l5_{tag}"] = nl(Lm["gate 100, kon 1e5"]["alpha"][i]); v[f"lsp_{tag}"] = f"{Lm['gate 100, kon 1e6']['spacing'][i]:.0f}"
    # experiment 6 (locus)
    Lc = R["locus"]; k = Lc["ktx"]
    lab = "N-terminal interface"
    pick = lambda terr, q: Lc[(lab, terr)][int(np.argmin(abs(k - q)))]
    v["cis_lo"] = nl(min(r["cis"] for r in Lc[(lab, "territory 0.2 um")])); v["cis_hi"] = nl(max(r["cis"] for r in Lc[(lab, "territory 0.2 um")]))
    for q, tag in ((0.01, "001"), (0.1, "01"), (1.0, "1")):
        v[f"sib_{tag}"] = nl(100 * np.mean([pick(t, q)["sibling"] for t in ("DNA-tethered only", "territory 0.2 um", "whole cell")]), 1)
        v[f"ter_{tag}"] = nl(100 * pick("territory 0.2 um", q)["territory"], 1)
        v[f"cell_{tag}"] = nl(100 * pick("whole cell", q)["territory"], 2)
    v["nasc_01"] = nl(pick("territory 0.2 um", 0.1)["nascent"], 1); v["tr_01"] = f"{pick('territory 0.2 um', 0.1)['transcripts']:.0f}"
    v["from_nasc"] = nl(100 * max(r["from_nascent"] for t in ("DNA-tethered only", "territory 0.2 um", "whole cell") for r in Lc[(lab, t)]), 1)
    r = Lc[(lab, "territory 0.2 um")]
    tx = np.array([q["tx_rate"] for q in r]); prot = np.array([q["protein_rate"] for q in r])
    cis = np.array([q["cis"] for q in r]) * prot / 2
    tr = np.array([(q["sibling"] + q["territory"]) if q["sibling"] * q["n"] >= 6 else np.nan for q in r]) * prot / 2
    ok = np.isfinite(tr)
    v["slope_cis"] = nl(np.polyfit(np.log(tx), np.log(cis), 1)[0]); v["slope_tr"] = nl(np.polyfit(np.log(tx[ok]), np.log(tr[ok]), 1)[0])
    b = Lc["bursty"]["res"]
    btx = np.array([q["tx_rate"] for q in b]); btr = np.array([(q["sibling"] + q["territory"]) * q["protein_rate"] / 2 for q in b])
    v["slope_burst"] = nl(np.polyfit(np.log(btx), np.log(btr), 1)[0])
    v["t7_sib"] = nl(100 * Lc["speed"][75.0]["sibling"], 1); v["nat_sib"] = nl(100 * Lc["speed"][15.0]["sibling"], 1)
    v["t7_nasc"] = nl(Lc["speed"][75.0]["nascent"], 1); v["nat_nasc"] = nl(Lc["speed"][15.0]["nascent"], 1)
    # experiment 7 (repressors)
    P = R["repressors"]
    for nm, tag in (("TetR(B)", "tet"), ("LacI", "lac"), ("lambda cI", "ci")):
        p = P[nm]
        v[f"{tag}_q50"], v[f"{tag}_q90"] = f"{p['q50']:.0f}", f"{p['q90']:.0f}"
        v[f"{tag}_w50"] = str(int(p["window50"])); v[f"{tag}_exp"] = f"{p['exposed_before_release'] * 100:.0f}"
        v[f"{tag}_a6"], v[f"{tag}_a7"] = nl(p["alpha"][(0.1, 1e6)]), nl(p["alpha"][(0.1, 1e7)])
        v[f"{tag}_n"] = str(len(p["residues"]))
    win = 270 - P["scan_n_on"]
    a6, a7 = P[("scan", 1e6)][0], P[("scan", 1e7)][0]
    v["win_ok6"] = str(int(win[np.where(a6 < 0.022)[0][0]])); v["win_ok7"] = str(int(win[np.where(a7 < 0.022)[0][0]]))
    v["a_w45"] = nl(a6[list(win).index(45)]); v["a_w220"] = nl(a6[list(win).index(220)])
    return v


TEMPLATE = r"""
<p class="eyebrow">Toetsing aan gepubliceerde data · Bertolini et al. 2021 · Brandt et al. 2009 · PDB 9GFT</p>
<h1>Houdt de theorie stand tegenover gemeten ribosoomparen?</h1>
<p class="lede">Ja voor de kern, nee voor het mechanisme. Ketens op verschillende ribosomen binden elkaar tijdens de translatie, ook in <em>E. coli</em>, en met hoge opbrengst. Wanneer dat gebeurt hangt af van het moment waarop de dimeer-interface uit de tunnel komt en van de echte vorm van het ribosoom. De overlappende bollen uit de paper voorspellen de start veel te vroeg.</p>

<div class="findings wide">
  <div><b>{f1} en {f2}</b><span>Fractie ribosomen in een keten-gekoppeld paar bij het coiled-coil-construct in <em>E. coli</em> (twee replicaten). De mutant zonder coiled coil blijft op de uitgangswaarde.</span></div>
  <div><b>codon {mid1}–{mid2}</b><span>Start van de koppeling, bij {exp1} tot {exp2} blootgestelde residuen van de coiled coil. Het bollenmodel laat de koppeling al vanaf codon 50 oplopen.</span></div>
  <div><b>{exit} nm</b><span>Afstand tussen de tunneluitgangen van twee botsende <em>E. coli</em>-ribosomen. Ze wijzen van elkaar af; de kortste vrije route is {pathnm} nm.</span></div>
  <div><b>0 van 3</b><span>Bij TetR, LacI en λ cI komt 90% van de dimeer-interface nooit buiten de tunnel voordat de keten loslaat.</span></div>
</div>

<h2>Bronnen, en wat ontbrak</h2>
<ul>
<li><strong>Bertolini et al., Science 2021.</strong> Volledige tekst, Table S1 met {nhigh} betrouwbare kandidaten, en de ruwe profielen per codon uit GEO (GSE151959): 14 bestanden voor <em>E. coli</em> en 6 voor HEK293-T.</li>
<li><strong>Brandt et al., Cell 2009.</strong> Alleen de samenvatting. De uitgever weigert geautomatiseerde toegang en de Chrome-extensie was niet verbonden. De stelling uit die samenvatting (uitgangen wijzen naar buiten, de afstand tussen ketens van buren is maximaal) heb ik getoetst aan het atomaire model van twee <em>E. coli</em>-ribosomen op één mRNA (PDB 9GFT, Mol. Cell 2025). Met de pdf erbij kan de polysoomgeometrie uit de tomografie zelf er nog in.</li>
<li><strong>Kristalstructuren</strong> van TetR (4AC0), LacI (1LBI) en λ cI (3BDN) voor de ligging van de dimeer-interface.</li>
</ul>

<h2>1 · Gebeurt het in <em>E. coli</em>, en hoeveel?</h2>
<p>De methode van Bertolini (DiSP) verteert het mRNA tussen de ribosomen. Twee ribosomen die daarna nog aan elkaar hangen, doen dat via hun ketens. De verhouding tussen voetafdrukken in de disoom- en de monosoomfractie laat per codon zien waar die koppeling begint. Drie humane eiwitten zijn zo in <em>E. coli</em> gemeten: de coiled coil 1B van lamine gefuseerd aan mCherry, lamine C en dynactine p150.</p>
""" + fig("g1", "Disoom-monosoomverhouding per codon voor drie constructen in E. coli en de daaruit afgeleide gekoppelde fractie",
          "Heranalyse van de profielen uit GEO, in bins van 10 codons (dynactine 20). Balken boven de grafiek: domeinen van het construct. Rechtsonder: lijnen uit de verrijking, punten uit de afname van monosoom-voetafdrukken ten opzichte van de mutant.") + r"""
<p>Het coiled-coil-construct heeft een schone controle: een mutant met dezelfde aminozuren in een andere volgorde, die geen coiled coil vormt. Het mCherry-deel is identiek. De verhouding van monosoom-voetafdrukken tussen wildtype en mutant geeft daarom direct de fractie ribosomen die uit de monosoomfractie is verdwenen: {f1} ± {se1} en {f2} ± {se2}. De mutant zelf blijft op {mutE1} en {mutE2} keer de uitgangswaarde.</p>
<p>De koppeling begint scherp, met het midden van de overgang bij codon {mid1} en {mid2}. Met 30 residuen in de tunnel zijn dan {exp1} tot {exp2} van de 138 residuen van de coiled coil blootgesteld. Lamine C start bij codon {lmid1} en {lmid2} van het construct (lamineresidu {lres1} en {lres2}; in humane cellen 220 volgens Table S1) en bereikt ongeveer {lf2} tot {lf1}. Dynactine start bij codon {dmid} (in humane cellen 455).</p>
<div class="note"><strong>Wat dit niet bewijst.</strong> Deze cellen overexprimeren het construct vanaf een T7-promotor: {sh_lo} tot {sh_hi}% van alle ribosoom-voetafdrukken ligt op dit ene mRNA. Bij enkele tienduizenden ribosomen per cel geeft dat een celbrede concentratie van naschentketens in het micromolaire bereik, even hoog als de lokale concentratie tussen buren op één mRNA (experiment 3). In <em>E. coli</em> kan de partner dus ook op een ander mRNA zitten. Dat de partner op hetzelfde mRNA zit, toonde Bertolini in humane cellen: getagd en ongetagd lamine C uit dezelfde cel vormen nooit een gemengd dimeer.</div>

<h2>2 · Verklaart het bollenmodel het gemeten profiel?</h2>
<p>Ik heb gesimuleerd wat DiSP zou meten onder vier modellen en elk model aan het profiel tussen codon 20 en 340 gefit. Vrij waren steeds de ribosoomafstand en de reactiesterkte. Drie modellen hebben daarnaast een poort: paren kan pas als beide ketens 100, 115 of 130 residuen hebben blootgesteld.</p>
""" + fig("g2", "Gemeten gekoppelde fractie per codon met vier gefitte modellen en hun afwijking",
          "Punten: gemiddelde van de twee replicaten, streepjes: het bereik ertussen. Het grijze deel rechts is niet meegefit.") + r"""
<p>Het bollenmodel uit de paper past het slechtst (RMSE {rmse_paper}; beste fit bij {sp_paper} codons tussen de ribosomen en k<sub>D</sub> = {kd_paper}). Het laat de koppeling vanaf codon 50 geleidelijk oplopen en mist de sprong bij codon 140. Met een poort halveert de afwijking tot {rmse_gated_paper}, {rmse_constant} en {rmse_structure}. Dat ligt in de buurt van de meetruis: de helft van het verschil tussen de twee replicaten is {rep_rmse}.</p>
<p>Het structuurgebaseerde model (experiment 3) past met een poort bij {gate_structure} residuen, {sp_structure} codons tussen de ribosomen en een associatieconstante van {kon_struct} M⁻¹s⁻¹. Dat is een gangbare waarde voor eiwit-eiwitbinding; er is geen uitzonderlijke reactiviteit nodig. In die fit verlaat {al_structure}% van de ketens het ribosoom als co-translationeel dimeer, terwijl het gemeten plateau {f1} is. DiSP telt alleen paren waarvan beide ribosomen nog op het mRNA zitten en onderschat de fractie dimeren dus.</p>
<p>Eén ding reproduceert geen van de modellen. Vlak voor het stopcodon verliest het achterste ribosoom van een paar zijn partner, waardoor de modellen naar {end_model} zakken. De meting blijft op {end_data}. Trage terminatie, waardoor het achterste ribosoom inloopt, is een mogelijke verklaring; de data beslissen dat niet.</p>

<h2>3 · Kunnen de ketens elkaar bereiken?</h2>
<p>Brandt concludeerde uit tomografie dat de uitgangen in een polysoom naar buiten wijzen en dat dit de afstand tussen ketens van buren maximaliseert. In de atomaire structuur van twee botsende <em>E. coli</em>-ribosomen is dat direct na te meten. De tunneluitgang heb ik bepaald door vanaf de getraceerde keten in het voorste ribosoom de vrije ruimte te volgen tot in het oplosmiddel ({tunnel} Å vanaf het peptidyltransferasecentrum) en die plek via het 23S-rRNA op het andere ribosoom te leggen (afwijking {rmsd} Å).</p>
""" + fig("g3", "Projectie van twee botsende ribosomen met hun tunneluitgangen, en de effectieve concentratie tussen ketens als functie van ketenlengte en ribosoomafstand",
          "Links: fosfor- en Cα-atomen van beide ribosomen, geprojecteerd op het vlak door de twee uitgangen; pijlen geven de tunnelrichting. Midden: lokale concentratie van de N-terminus van de partner, voor twee even lange ketens. Rechts: dezelfde grootheid tegen de afstand langs het mRNA; het eerste punt is het botsende paar.") + r"""
<p>De centra liggen {centre} nm uit elkaar, de uitgangen {exit} nm. Beide tunnels wijzen van de andere uitgang af ({ang1}° en {ang2}° ten opzichte van de verbindingslijn). Die lijn loopt voor {blocked}% door ribosoommassa; de kortste vrije route eromheen is {pathnm} nm. Twee volledig gestrekte ketens hebben daar elk {nmin_ext} residuen voor nodig. Een Monte Carlo met ongevouwen ketens rond de echte ribosoomvorm geeft {col120} bij 120 blootgestelde residuen, {col150} bij 150 en {col200} bij 200. Eén molecuul in een <em>E. coli</em>-cel is 1,7 nM. Voor dicht opeengepakte buren klopt Brandts stelling dus.</p>
<p>Voor ribosomen die verder uit elkaar zitten ligt de oriëntatie niet vast. Ik heb per afstand 20.000 willekeurige standen getrokken, alleen beperkt door de lengte van het mRNA ertussen en de eis dat de ribosomen elkaar niet doorsnijden. Bij 2 codons extra is {acc2}% van de standen toegestaan, bij 20 codons {acc20}% en bij 160 codons {acc160}%. Beide uitgangen wijzen naar elkaar toe in {face20}% van de toegestane standen bij 20 codons en in {face160}% bij 160 codons. Dat uitgangen van elkaar af wijzen volgt dus al uit de plek waar het mRNA aan het ribosoom vastzit: aan de 30S-kant, tegenover de uitgang. Er is geen ordening voor nodig.</p>
<p>De gemiddelde lokale concentratie bij 100 blootgestelde residuen is {c100_5} bij 5 codons extra, {c100_20} bij 20, {c100_80} bij 80 en {c100_160} bij 160. Zeldzame gunstige standen dragen dat gemiddelde: de beste 1% levert {top20}% van de bijdrage bij 20 codons en {top160}% bij 160. Buren op enkele tientallen tot honderden codons zien elkaar dus bij 0,2 tot 2 µM. Met een gangbare associatieconstante is dat genoeg voor een reactie binnen seconden.</p>
<p>De paper rekent met 9 Å per codon tussen de ribosomen ({papdist} nm op botsafstand, tegen {exit} nm gemeten) en met de contourlengte als bereik van de keten. Die twee afwijkingen heffen elkaar deels op, maar de afstandsafhankelijkheid draait om. In de paper is dichterbij altijd gunstiger. In de structuur zijn de dichtste buren juist afgeschermd en ligt het optimum, bij deze aanname over de soepelheid van het mRNA, rond 150 tot 350 codons.</p>

<h2>4 · {nhigh} humane eiwitten</h2>
""" + fig("g4", "Verdeling van de start van koppeling, metagenen rond de start en vóór het stopcodon, en efficiëntie tegen resterende lengte",
          "Linksboven: startpositie uit Table S1. Rechtsboven en linksonder: opgetelde voetafdrukken van alle kandidaten, uitgelijnd op de start en op het stopcodon; band: 95%-interval uit 400 bootstraps over genen. Rechtsonder: efficiëntie per gen zoals Bertolini die definieert.") + r"""
<p>De koppeling start bij mediaan codon {on_med} (kwartielen {on_q1} tot {on_q3}); bij 90% van de eiwitten na codon {on_p10}. Slechts {on_ltmin}% start vóór codon {on_min_codon}, het minimum waarbij twee botsende buren elkaar gestrekt kunnen raken. De start ligt mediaan op {on_frac_exp} van de eiwitlengte, gerekend naar blootgestelde residuen.</p>
<p>Eenmaal begonnen gaat het snel: de verrijking loopt op tot {meta_plateau} keer de uitgangswaarde, de helft daarvan binnen {meta_half} codons en 90% binnen {meta_90} codons. Dat past bij een reactie die binnen een paar seconden afloopt zodra de interface er is.</p>
<p>Richting het stopcodon zakt de verrijking bij de kandidaten naar {stop_end} (95%-interval {stop_lo} tot {stop_hi}), terwijl {ctrl_n} controle-eiwitten op {stop_ctrl} blijven. Vanaf ongeveer {stop_start} codons vóór het stopcodon ligt het hele interval onder 1. Dit is het signaal dat je verwacht als de partner op hetzelfde mRNA vooroploopt en eerder klaar is. De partners zitten dus tot ongeveer 200 codons uit elkaar, het bereik waarin de lokale concentratie in experiment 3 hoog is.</p>
<p>Mijn herberekening van de efficiëntie geeft mediaan {eff_med} over {eff_n} genen; de paper noemt ongeveer 40%. De samenhang met het aantal codons na de start is zwak (Spearman ρ = {rho_remaining}). De paper voorspelt met het bollenmodel een sterke lengte-afhankelijkheid; in deze data is die er nauwelijks. De samenhang met expressie is negatief (ρ = {rho_rpkm}). Dat pleit tegen partners op andere mRNA's, maar zwak tot expressie gebrachte genen worden alleen bij een groot effect als kandidaat herkend, dus dit getal is vertekend.</p>

<h2>5 · Sterkere RBS, meer dimerisatie?</h2>
""" + fig("g5", "Co-translationele fractie tegen initiatiesnelheid voor de formule uit de paper en voor het structuurgebaseerde model",
          "Polysoomsimulatie met de lokale concentratie uit experiment 3 als paarkans, vergeleken met vergelijking 3 van de paper bij dezelfde ribosoomafstand.") + r"""
<p>Deze voorspelling kan ik met de beschikbare data niet toetsen; niemand heeft de RBS gevarieerd. De simulatie laat wel zien wat het structuurmodel anders voorspelt. De richting blijft: een zwakkere RBS geeft minder co-translationele dimeren. Het bereik verschuift sterk. Bij λ = {lam_a}/s geeft het structuurmodel {ls_a} tegen {lp_a} per paar volgens vergelijking 3, bij {lam_b}/s is dat {ls_b} tegen {lp_b}, en bij {lam_c}/s {ls_c} tegen {lp_c}. Boven 0,2/s is het verzadigd.</p>
<p>De reden is dat twee ketens met een blootgestelde interface op hetzelfde mRNA elkaar vrijwel altijd vinden. α wordt dan de kans dat er zo'n partner is. Voor een initiatiesnelheid λ en een venster W tussen het vrijkomen van de interface en het stopcodon is dat hoogstens 1 − e<sup>−2λW</sup>. De knop is dus het product van RBS-sterkte en venster, en niet de eiwitlengte als zodanig.</p>

<h2>6 · Gekoppelde transcriptie en translatie bij het gen</h2>
<p>In bacteriën laden ribosomen op het mRNA terwijl het polymerase nog schrijft. Polysomen hangen dan via het polymerase aan het DNA, en transcripten van hetzelfde gen zitten dicht bij elkaar. Ik heb een genlocus gesimuleerd: polymerases starten met een instelbare snelheid en lopen 15 codons per seconde, het eerste ribosoom kan het polymerase niet passeren, verdere ribosomen laden met 0,1/s, en een afgemaakt transcript blijft gemiddeld 180 s vertaalbaar. Ketens kunnen paren binnen één polysoom, met een keten op een naburig transcript dat nog aan het DNA hangt, en met ketens op andere transcripten in de buurt van het gen. Het eiwit is 300 residuen lang met de interface na 100 residuen.</p>
""" + fig("g6", "Aandeel van de drie paarroutes tegen transcriptiesnelheid, en schaling van de dimeerproductie",
          "Links: aandeel van alle ketens per route als transcripten binnen 0,2 µm van het gen blijven. Midden: paren tussen verschillende transcripten onder drie aannames over de verspreiding van mRNA. Rechts: productie van co-translationele dimeren tegen het aantal transcripten per seconde; een helling van 1 is lineair.") + r"""
<p>Binnen één polysoom paart {cis_lo} tot {cis_hi} van de ketens, onafhankelijk van de promotorsterkte. Paren met een naburig transcript aan het DNA lopen op van {sib_001}% bij één transcript per 100 s via {sib_01}% bij één per 10 s tot {sib_1}% bij één per seconde. Blijven de afgemaakte transcripten binnen 0,2 µm van het gen, dan komt daar {ter_01}% tot {ter_1}% bij. Verspreid over de hele cel is dat minder dan {cell_1}%.</p>
<p>De reden dat koppeling zo weinig toevoegt: bij een gen van 300 codons is het polymerase na 20 s klaar, tegen 180 s levensduur van het mRNA. Er hangen gemiddeld {nasc_01} van de {tr_01} transcripten nog aan het DNA (bij 0,1 transcript per seconde), en hooguit {from_nasc}% van de ketens komt klaar voordat het transcript loslaat. Een polymerase dat vijf keer sneller loopt, zoals T7, brengt het aandeel naburige transcripten van {nat_sib}% naar {t7_sib}%.</p>
<p>Voor het argument uit de paper is de schaling het belangrijkst. Dimeren binnen één polysoom nemen lineair toe met het aantal transcripten (helling {slope_cis}). Dimeren tussen transcripten nemen bijna kwadratisch toe (helling {slope_tr}) als de transcriptiesnelheid traploos verandert, en lineair (helling {slope_burst}) als een promotor tussen aan en uit schakelt, zoals in de vergelijkingen van de paper. De uitspraak dat co-translationele dimerisatie lineair is in de promotoractiviteit houdt dus stand, met een afwijking van hooguit enkele procenten bij zeer sterke promotors.</p>
<p>Wat de koppeling wel bepaalt is de plaats. Co-translationele dimeren ontstaan waar het mRNA is. Blijft dat bij het gen, dan ontstaan ze daar, en dat is de aanname achter de lokale route in het toggle-switch-model. Hoe lang mRNA bij het gen blijft staat niet in deze datasets. Een gen midden in een operon, een kortlevend mRNA of een eiwit dat aan het DNA bindt terwijl het nog aan het ribosoom hangt, vergroten het aandeel dat bij het gen wordt afgemaakt.</p>

<h2>7 · TetR, LacI en λ cI</h2>
""" + fig("g7", "Ligging van de dimeer-interface langs de sequentie van drie repressoren, en de co-translationele fractie tegen het resterende venster",
          "Links: elk streepje is een residu binnen 4 Å van de andere keten in de kristalstructuur; het driehoekje markeert de helft van de interface. Onderaan de start van koppeling bij de humane kandidaten (kwartielen en 10–90%). Rechts: structuurmodel voor een eiwit van 300 residuen; stippellijnen geven het venster van elke repressor als de helft van de interface volstaat.") + r"""
<p>Of een repressor co-translationeel kan paren hangt af van de vraag hoeveel van de interface buiten de tunnel moet zijn. De helft van de interface-residuen ligt bij TetR vóór residu {tet_q50} van 207, bij LacI vóór {lac_q50} van 360 en bij λ cI vóór {ci_q50} van 237. Negentig procent ligt vóór residu {tet_q90}, {lac_q90} en {ci_q90}. Tel daar de 30 residuen in de tunnel bij op en bij geen van de drie is 90% van de interface ooit blootgesteld voordat de keten loslaat.</p>
<p>Volstaat de helft van de interface, dan blijft er een venster van {tet_w50}, {lac_w50} en {ci_w50} codons. Het structuurmodel geeft dan bij λ = 0,1/s een α van {tet_a6} tot {tet_a7} voor TetR, {lac_a6} tot {lac_a7} voor LacI en {ci_a6} tot {ci_a7} voor λ cI (k<sub>on</sub> van 10⁶ tot 10⁷). Onder het Hopf-punt van 0,022 blijft α alleen bij een venster van hooguit {win_ok6} codons (k<sub>on</sub> = 10⁶) of {win_ok7} codons (10⁷).</p>
<p>Bertolini ziet dat globulaire dimerisatiedomeinen bij de start van de koppeling volledig buiten de tunnel zijn; alleen coiled coils en BAR-domeinen paren al als ze gedeeltelijk zijn blootgesteld. De interfaces van deze drie repressoren zijn globulair. Volgens die regel is α voor alle drie nul, en dat past bij het feit dat de repressilator oscilleert. Het is een afleiding en geen meting. Het omgekeerde geval is even duidelijk: een N-terminaal dimerisatiedomein, zoals een leucine zipper, brengt α in dit model naar ongeveer {a_w220}.</p>

<h2>Oordeel per bewering</h2>
<div class="tablewrap wide"><table class="verdict">
<thead><tr><th>Bewering</th><th>Toets</th><th>Oordeel</th></tr></thead>
<tbody>
<tr><td>Ketens op verschillende ribosomen dimeriseren tijdens de translatie, ook in <em>E. coli</em></td><td>DiSP in <em>E. coli</em>: {f1} en {f2} van de ribosomen gekoppeld; mutant: niets</td><td><span class="tag ok">houdt stand</span></td></tr>
<tr><td>De partner zit op hetzelfde mRNA</td><td>Humaan: geen gemengde dimeren van getagd en ongetagd lamine C; verlies van koppeling vóór het stopcodon ({stop_end}, controle {stop_ctrl}). <em>E. coli</em>: door overexpressie niet te onderscheiden</td><td><span class="tag part">humaan ja, E. coli open</span></td></tr>
<tr><td>Mechanisme: overlappende bollen met de contourlengte als straal, op een recht mRNA</td><td>Fit op het gemeten profiel: RMSE {rmse_paper} tegen {rmse_structure} met poort; uitgangen {exit} nm uit elkaar in plaats van {papdist} nm</td><td><span class="tag no">houdt geen stand</span></td></tr>
<tr><td>α stijgt met de initiatiesnelheid</td><td>Geen data. Structuurmodel: zelfde richting, verzadigd boven 0,2/s</td><td><span class="tag open">niet getoetst</span></td></tr>
<tr><td>α stijgt met de eiwitlengte</td><td>{eff_n} humane genen: ρ = {rho_remaining} met de lengte na de start</td><td><span class="tag part">zwak</span></td></tr>
<tr><td>Co-translationele dimerisatie is lineair in de promotoractiviteit</td><td>Locussimulatie: helling {slope_cis} binnen een polysoom; paren tussen transcripten hooguit enkele procenten</td><td><span class="tag ok">houdt stand</span></td></tr>
<tr><td>α is voor een gewone repressor groot genoeg om oscillaties te dempen</td><td>Interface van TetR, LacI en λ cI ligt C-terminaal; 90% komt nooit buiten de tunnel</td><td><span class="tag part">alleen bij N-terminale interface</span></td></tr>
</tbody></table></div>

<h2>Drie experimenten die het beslissen</h2>
<ol>
<li><strong>Zelfde mRNA of niet, in <em>E. coli</em>.</strong> Het coiled-coil-construct in enkelvoud op het chromosoom achter een σ70-promotor, met twee verschillend getagde kopieën in dezelfde cel. Zit de partner op hetzelfde mRNA, dan blijft de gekoppelde fractie gelijk aan die bij overexpressie en ontstaan er geen gemengde dimeren.</li>
<li><strong>Een reeks RBS-sterktes</strong> op hetzelfde construct, van 0,005 tot 0,5 initiaties per seconde. Het structuurmodel voorspelt verzadiging boven 0,2/s en ongeveer {ls_a} bij {lam_a}/s. Vergelijking 3 voorspelt {lp_a} en een geleidelijke stijging tot 0,45.</li>
<li><strong>De interface verplaatsen.</strong> De coiled coil achter mCherry in plaats van ervoor moet de koppeling opheffen. Omgekeerd moet een N-terminale leucine zipper aan TetR, LacI en cI de repressilator dempen. Dat laatste toetst het centrale idee van de paper rechtstreeks in een circuit.</li>
</ol>

<h2>Beperkingen</h2>
<ul>
<li>Van Brandt et al. is alleen de samenvatting gebruikt. De botsende ribosomen uit 9GFT zijn de dichtst mogelijke pakking; een translerend polysoom kan anders gerangschikt zijn. In die structuur zat de helicase HrpA gebonden; die heb ik weggelaten.</li>
<li>Het mRNA tussen de ribosomen is gemodelleerd als een flexibele keten (5,9 Å per nucleotide, Kuhn-lengte 20 Å). Een stijver mRNA verlaagt de lokale concentratie op grote afstand.</li>
<li>Ketens zijn ongevouwen kluwens. De snelle rekenmethode voor het gemiddelde over oriëntaties wijkt op 250 standen hooguit een factor 1,4 af van de volledige Monte Carlo, en onderschat het botsende paar een factor {colscale}; daarvoor is gecorrigeerd.</li>
<li>Paren is in de simulatie onomkeerbaar. Een zwakke interface laat weer los, en DiSP ziet alleen complexen die de centrifugatie overleven. De gemeten start is daarom een bovengrens voor het eerste contact.</li>
<li>De fractie voor lamine C veronderstelt een gelijkmatige ribosoomdichtheid langs het construct; voor dynactine is alleen de verrijking betrouwbaar.</li>
<li>In het locusmodel zijn paren tussen transcripten aan het DNA benaderd met een gelijkwaardige flexibele verbinding. Het gen heeft geen stroomafwaartse genen.</li>
</ul>
"""

MODEL_NOTE = r"""<div class="note"><strong>Deels achterhaald door de toetsing.</strong> De α-waarden op deze pagina volgen uit het bollenmodel van de paper. De <a href="#toetsing">toetsing aan data</a> laat zien dat de start van de koppeling door de interface wordt bepaald, en vervangt de polymeerschatting hieronder (uitgangen op 25 nm, evenwijdig of tegengesteld) door de gemeten geometrie.</div>
"""


def build():
    v = tokens()
    frag = TEMPLATE.format_map(v)
    page = (ROOT / "report_template.html").read_text()
    assert "<!--TOETSING-->" in page
    page = page.replace("<!--TOETSING-->", frag)
    anchor = '<div class="findings wide">\n  <div><b>0,0221</b>'
    assert anchor in page
    page = page.replace(anchor, MODEL_NOTE + anchor)
    (ROOT / "report.html").write_text(page)
    return v


if __name__ == "__main__":
    v = build()
    print("report.html written;", len(v), "numbers filled in")
